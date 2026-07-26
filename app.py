from flask import (
    Flask,
    abort,
    flash,
    g,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from dotenv import load_dotenv
from openai import OpenAI
from datetime import datetime, timedelta
from functools import wraps
import os
from pathlib import Path
import json
import re
import secrets
import sqlite3
import subprocess
import tempfile
import unicodedata
from urllib.parse import urlsplit
import uuid
from zoneinfo import ZoneInfo

from werkzeug.security import check_password_hash, generate_password_hash

import database
from database import (
    bind_elder,
    can_message_contact,
    count_unread_messages,
    count_assessments,
    create_message,
    create_user,
    get_assessment,
    get_user_by_id,
    get_user_by_username,
    is_elder_bound_to,
    list_assessments,
    list_bound_elders,
    list_elder_caregivers,
    list_message_contacts,
    list_messages,
    list_users_with_stats,
    mark_messages_read,
    save_assessment,
    unbind_elder,
    update_last_login,
    update_user_active,
    update_user_password,
)
from questions import (
    QUESTIONS,
    domain_max_points,
    result_level,
    score_answer,
    total_points,
)
from tencent_asr import TencentASRError, recognize_pcm
from reports import build_assessment_report

app = Flask(__name__)
load_dotenv()
app.secret_key = os.getenv("FLASK_SECRET_KEY", "development-only-change-me")
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
app.config["DATABASE_PATH"] = os.getenv("DATABASE_PATH")
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (
    os.getenv("SESSION_COOKIE_SECURE", "false").lower() == "true"
)

# ================= 配置区域 =================
# 1. DeepSeek 大模型配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
TENCENT_ASR_APP_ID = os.getenv("TENCENT_ASR_APP_ID")
TENCENT_ASR_SECRET_ID = os.getenv("TENCENT_ASR_SECRET_ID")
TENCENT_ASR_SECRET_KEY = os.getenv("TENCENT_ASR_SECRET_KEY")
TENCENT_ASR_ENGINE = os.getenv("TENCENT_ASR_ENGINE", "16k_zh_en")
TENCENT_ASR_ENGINES = {
    "chaoshan": os.getenv(
        "TENCENT_ASR_ENGINE_CHAOSHAN",
        TENCENT_ASR_ENGINE,
    ),
    "cantonese": os.getenv(
        "TENCENT_ASR_ENGINE_CANTONESE",
        "16k_yue",
    ),
}
SPEECH_LANGUAGE_LABELS = {
    "chaoshan": "潮汕话",
    "cantonese": "粤语",
}
FFMPEG_PATH = Path(__file__).with_name("ffmpeg.exe")

# ===========================================

database.init_app(app)

# 原型阶段将测试进度保存在服务进程内，避免把老人回答写入浏览器 Cookie。
TEST_SESSIONS = {}

# 初始化 DeepSeek 客户端。DeepSeek 兼容 OpenAI Chat Completions 接口。
deepseek_client = OpenAI(
    api_key=DEEPSEEK_API_KEY or "missing-deepseek-api-key",
    base_url=DEEPSEEK_BASE_URL,
    timeout=60.0,
)


def normalize_username(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def password_error(password, confirmation=None):
    if confirmation is not None and password != confirmation:
        return "两次输入的密码不一致。"
    if len(password) < 8:
        return "密码至少需要8位。"
    if len(password) > 128:
        return "密码不能超过128位。"
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return "密码必须同时包含英文字母和数字。"
    return None


def is_safe_next_url(target):
    if not target:
        return False
    parsed = urlsplit(target)
    return not parsed.scheme and not parsed.netloc and target.startswith("/")


def csrf_token():
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def is_api_request():
    return (
        request.path == "/chat"
        or request.path.startswith("/test/")
        or request.path.startswith("/speech/")
    )


@app.before_request
def load_current_user_and_protect_posts():
    user_id = session.get("user_id")
    g.current_user = get_user_by_id(user_id) if user_id else None

    if g.current_user is not None and not g.current_user["is_active"]:
        session.clear()
        g.current_user = None

    if request.method == "POST":
        supplied_token = (
            request.headers.get("X-CSRF-Token")
            or request.form.get("_csrf_token")
        )
        expected_token = session.get("_csrf_token")
        if (
            not expected_token
            or not supplied_token
            or not secrets.compare_digest(expected_token, supplied_token)
        ):
            if is_api_request():
                return jsonify({"error": "页面验证已失效，请刷新页面后重试。"}), 400
            flash("页面验证已失效，请重新操作。", "error")
            return redirect(request.referrer or url_for("login"))
    return None


@app.context_processor
def inject_template_context():
    return {
        "current_user": g.get("current_user"),
        "csrf_token": csrf_token,
        "unread_message_count": (
            count_unread_messages(g.current_user["id"])
            if g.get("current_user")
            and g.current_user["account_type"] in {"family", "community"}
            else 0
        ),
    }


def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if g.current_user is None:
            if is_api_request():
                return jsonify({"error": "请先登录后再使用。"}), 401
            return redirect(url_for("login", next=request.full_path.rstrip("?")))
        return view(*args, **kwargs)

    return wrapped_view


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        if g.current_user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)

    return wrapped_view


def caregiver_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        if g.current_user["account_type"] not in {"family", "community"}:
            abort(403)
        return view(*args, **kwargs)

    return wrapped_view


def can_view_elder(elder_id):
    if g.current_user["role"] == "admin":
        return True
    if (
        g.current_user["account_type"] == "elder"
        and g.current_user["id"] == elder_id
    ):
        return True
    return (
        g.current_user["account_type"] in {"family", "community"}
        and is_elder_bound_to(g.current_user["id"], elder_id)
    )


@app.template_filter("local_datetime")
def local_datetime(value):
    if not value:
        return "暂无"
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(ZoneInfo("Asia/Shanghai")).strftime(
        "%Y年%m月%d日 %H:%M"
    )


@app.template_filter("language_name")
def language_name(value):
    return {
        "browser": "普通话",
        "cantonese": "粤语",
        "chaoshan": "潮汕话",
    }.get(value, value or "未记录")


@app.template_filter("account_type_name")
def account_type_name(value):
    return {
        "elder": "老人端",
        "family": "家属端",
        "community": "社区端",
        "admin": "后台管理员",
    }.get(value, value or "未设置")


def get_ai_reply(user_text, system_prompt):
    """调用 DeepSeek，并返回最终回复文本。"""
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请检查项目根目录下的 .env 文件")

    completion = deepseek_client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        stream=False,
        # V4 默认开启思考；普通聊天关闭思考可减少等待时间。
        extra_body={"thinking": {"type": "disabled"}},
    )
    return completion.choices[0].message.content


def public_question(index):
    question = QUESTIONS[index]
    return {
        "number": index + 1,
        "total": len(QUESTIONS),
        "prompt": question["prompt"],
        "domain": question["domain"],
    }


def question_text(index):
    question = public_question(index)
    return (
        f"第 {question['number']}/{question['total']} 题"
        f"（{question['domain']}）：\n{question['prompt']}"
    )


def get_test_state():
    test_id = session.get("test_id")
    if not test_id:
        return None
    state = TEST_SESSIONS.get(test_id)
    if (
        state is None
        or g.current_user is None
        or state.get("user_id") != g.current_user["id"]
    ):
        return None
    return state


def format_result(state):
    maximum = total_points()
    percentage = round(state["score"] / maximum * 100)
    domain_maximums = domain_max_points()
    domain_lines = [
        f"{domain}：{state['domain_scores'].get(domain, 0)}/{points}"
        for domain, points in domain_maximums.items()
    ]
    return (
        "测试完成。\n\n"
        f"总分：{state['score']}/{maximum}（{percentage}分）\n\n"
        + "\n".join(domain_lines)
        + f"\n\n表现反馈：{result_level(percentage)}。\n\n"
        "提示：本结果仅反映本次任务表现，不构成认知障碍或其他疾病诊断。"
        "听力、情绪、疲劳、教育程度和语音识别准确率都可能影响结果。"
    )


@app.route("/")
def home():
    if g.current_user is None:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.current_user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = normalize_username(request.form.get("username"))
        password = str(request.form.get("password") or "")
        user = get_user_by_username(username)

        if (
            user is None
            or not user["is_active"]
            or not check_password_hash(user["password_hash"], password)
        ):
            flash("账号或密码不正确，请重新输入。", "error")
        else:
            remember = request.form.get("remember") == "on"
            next_url = request.form.get("next")
            session.clear()
            session["user_id"] = user["id"]
            session.permanent = remember
            update_last_login(user["id"])
            csrf_token()
            flash(f"{user['display_name']}，欢迎回来。", "success")
            if is_safe_next_url(next_url):
                return redirect(next_url)
            return redirect(url_for("dashboard"))

    return render_template(
        "login.html",
        next_url=request.args.get("next", ""),
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.current_user is not None:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        username = normalize_username(request.form.get("username"))
        display_name = unicodedata.normalize(
            "NFKC",
            str(request.form.get("display_name") or ""),
        ).strip()
        password = str(request.form.get("password") or "")
        confirmation = str(request.form.get("password_confirm") or "")

        error = None
        if not re.fullmatch(r"[A-Za-z0-9_\u4e00-\u9fff]{3,24}", username):
            error = "账号需为3至24位，只能使用中文、英文字母、数字或下划线。"
        elif get_user_by_username(username) is not None:
            error = "这个账号名称已经被使用，请换一个。"
        elif len(display_name) > 30:
            error = "称呼不能超过30个字。"
        else:
            error = password_error(password, confirmation)

        if error:
            flash(error, "error")
        else:
            account_type = str(
                request.form.get("account_type") or "elder"
            ).strip()
            if account_type not in {"elder", "family", "community"}:
                flash("请选择正确的账户类型。", "error")
                return render_template("register.html")
            try:
                create_user(
                    username=username,
                    password_hash=generate_password_hash(password),
                    display_name=display_name or username,
                    account_type=account_type,
                    role="member",
                )
            except sqlite3.IntegrityError:
                flash("这个账号名称已经被使用，请换一个。", "error")
            else:
                flash("注册成功，请使用新账号登录。", "success")
                return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout", methods=["POST"])
@login_required
def logout():
    test_id = session.get("test_id")
    if test_id:
        TEST_SESSIONS.pop(test_id, None)
    session.clear()
    flash("您已经安全退出。", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    if g.current_user["account_type"] in {"family", "community"}:
        return render_template(
            "dashboard.html",
            elders=list_bound_elders(g.current_user["id"]),
            contacts=list_message_contacts(g.current_user["id"]),
        )

    latest = list_assessments(g.current_user["id"], limit=1)
    return render_template(
        "dashboard.html",
        latest=latest[0] if latest else None,
        assessment_count=count_assessments(g.current_user["id"]),
        caregivers=(
            list_elder_caregivers(g.current_user["id"])
            if g.current_user["account_type"] == "elder"
            else []
        ),
    )


@app.route("/assistant")
@login_required
def assistant_page():
    if g.current_user["account_type"] in {"family", "community"}:
        abort(403)
    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    if g.current_user["account_type"] in {"family", "community"}:
        return redirect(url_for("dashboard"))
    return render_template(
        "history.html",
        owner=g.current_user,
        assessments=list_assessments(g.current_user["id"]),
        admin_view=False,
        caregiver_view=False,
    )


@app.route("/elders/<int:elder_id>/history")
@login_required
def elder_history(elder_id):
    elder = get_user_by_id(elder_id)
    if elder is None or elder["account_type"] != "elder":
        abort(404)
    if not can_view_elder(elder_id):
        abort(403)
    return render_template(
        "history.html",
        owner=elder,
        assessments=list_assessments(elder_id),
        admin_view=False,
        caregiver_view=True,
    )


@app.route("/history/<int:assessment_id>")
@login_required
def assessment_detail(assessment_id):
    assessment = get_assessment(assessment_id)
    if assessment is None:
        abort(404)
    if not can_view_elder(assessment["user_id"]):
        abort(403)
    return render_template(
        "assessment_detail.html",
        assessment=assessment,
        domain_scores=json.loads(assessment["domain_scores"]),
        domain_maximums=domain_max_points(),
    )


@app.route("/history/<int:assessment_id>/report")
@login_required
def assessment_report(assessment_id):
    assessment = get_assessment(assessment_id)
    if assessment is None:
        abort(404)
    if not can_view_elder(assessment["user_id"]):
        abort(403)

    scores = json.loads(assessment["domain_scores"])
    maximums = domain_max_points()
    return render_template(
        "assessment_report.html",
        assessment=assessment,
        domain_scores=scores,
        domain_maximums=maximums,
        report=build_assessment_report(assessment, scores, maximums),
    )


@app.route("/admin/users")
@admin_required
def admin_users():
    return render_template(
        "admin_users.html",
        users=list_users_with_stats(),
    )


@app.route("/admin/users/<int:user_id>/history")
@admin_required
def admin_user_history(user_id):
    user = get_user_by_id(user_id)
    if user is None:
        abort(404)
    return render_template(
        "history.html",
        owner=user,
        assessments=list_assessments(user_id),
        admin_view=True,
        caregiver_view=False,
    )


@app.route("/bindings", methods=["POST"])
@caregiver_required
def add_binding():
    code = unicodedata.normalize(
        "NFKC",
        str(request.form.get("binding_code") or ""),
    ).strip().upper()
    elder, created = bind_elder(g.current_user["id"], code)
    if elder is None:
        flash("没有找到这个老人绑定码，请核对后重新输入。", "error")
    elif created:
        flash(f"已成功绑定“{elder['display_name']}”。", "success")
    else:
        flash(f"您已经绑定过“{elder['display_name']}”。", "success")
    return redirect(url_for("dashboard"))


@app.route("/bindings/<int:elder_id>/remove", methods=["POST"])
@caregiver_required
def remove_binding(elder_id):
    if unbind_elder(g.current_user["id"], elder_id):
        flash("已解除绑定。历史结果访问权限也已同步取消。", "success")
    else:
        flash("没有找到这条绑定关系。", "error")
    return redirect(url_for("dashboard"))


@app.route("/messages", methods=["GET", "POST"])
@caregiver_required
def message_center():
    if request.method == "POST":
        contact_value = str(request.form.get("contact") or "")
        body = unicodedata.normalize(
            "NFKC",
            str(request.form.get("body") or ""),
        ).strip()
        try:
            recipient_text, elder_text = contact_value.split(":", 1)
            recipient_id = int(recipient_text)
            elder_id = int(elder_text)
        except (TypeError, ValueError):
            recipient_id = elder_id = 0

        if not body:
            flash("留言内容不能为空。", "error")
        elif len(body) > 1000:
            flash("每条留言不能超过1000个字。", "error")
        elif not can_message_contact(
            g.current_user["id"],
            recipient_id,
            elder_id,
        ):
            abort(403)
        else:
            create_message(
                elder_id,
                g.current_user["id"],
                recipient_id,
                body,
            )
            flash("留言已经发送。", "success")
            return redirect(url_for("message_center"))

    contacts = list_message_contacts(g.current_user["id"])
    messages = list_messages(g.current_user["id"])
    mark_messages_read(g.current_user["id"])
    return render_template(
        "messages.html",
        contacts=contacts,
        messages=messages,
    )


@app.route("/admin/users/<int:user_id>/toggle", methods=["POST"])
@admin_required
def admin_toggle_user(user_id):
    user = get_user_by_id(user_id)
    if user is None:
        abort(404)
    if user["id"] == g.current_user["id"]:
        flash("不能停用当前正在登录的管理员账号。", "error")
    else:
        new_status = not bool(user["is_active"])
        update_user_active(user_id, new_status)
        flash(
            f"已{'启用' if new_status else '停用'}账号“{user['username']}”。",
            "success",
        )
    return redirect(url_for("admin_users"))


@app.route(
    "/admin/users/<int:user_id>/reset-password",
    methods=["GET", "POST"],
)
@admin_required
def admin_reset_password(user_id):
    user = get_user_by_id(user_id)
    if user is None:
        abort(404)

    if request.method == "POST":
        password = str(request.form.get("password") or "")
        confirmation = str(request.form.get("password_confirm") or "")
        error = password_error(password, confirmation)
        if error:
            flash(error, "error")
        else:
            update_user_password(user_id, generate_password_hash(password))
            flash(f"账号“{user['username']}”的密码已重置。", "success")
            return redirect(url_for("admin_users"))

    return render_template("reset_password.html", user=user)


# 你原来的纯文本聊天接口（完全保持原样）
@app.route('/chat', methods=['POST'])
@login_required
def chat():
    user_input = request.json.get('message')

    if not user_input:
        return jsonify({'error': '没有收到消息'}), 400

    try:
        print("正在向 DeepSeek 提问:", user_input)
        ai_reply = get_ai_reply(
            user_input,
            "你是一个乐于助人的 AI 助手。",
        )
        return jsonify({'reply': ai_reply})

    except Exception as e:
        print("出错啦:", e)
        return jsonify({'reply': "抱歉，我出了一点小差错，请稍后再试。"}), 500


@app.route('/speech/recognize', methods=['POST'])
@login_required
def recognize_speech():
    if not all([
        TENCENT_ASR_APP_ID,
        TENCENT_ASR_SECRET_ID,
        TENCENT_ASR_SECRET_KEY,
    ]):
        return jsonify({
            "error": "腾讯云语音识别尚未配置，请填写 .env 中的腾讯云密钥。"
        }), 503

    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"error": "没有收到录音文件。"}), 400

    language = str(request.form.get("language", "chaoshan")).strip().lower()
    engine = TENCENT_ASR_ENGINES.get(language)
    if engine is None:
        return jsonify({"error": "不支持所选择的语音类型。"}), 400
    language_label = SPEECH_LANGUAGE_LABELS[language]

    if not FFMPEG_PATH.exists():
        return jsonify({"error": "项目目录中缺少 ffmpeg.exe。"}), 500

    try:
        with tempfile.TemporaryDirectory(prefix="memory_lantern_asr_") as temp_dir:
            input_path = Path(temp_dir) / "recording.webm"
            pcm_path = Path(temp_dir) / "recording.pcm"
            audio_file.save(input_path)

            conversion = subprocess.run(
                [
                    str(FFMPEG_PATH),
                    "-hide_banner",
                    "-loglevel", "error",
                    "-y",
                    "-i", str(input_path),
                    "-ar", "16000",
                    "-ac", "1",
                    "-f", "s16le",
                    str(pcm_path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            if conversion.returncode != 0 or not pcm_path.exists():
                print("音频转码失败:", conversion.stderr)
                return jsonify({"error": "录音格式转换失败，请重新录音。"}), 400

            transcript = recognize_pcm(
                pcm_path,
                TENCENT_ASR_APP_ID,
                TENCENT_ASR_SECRET_ID,
                TENCENT_ASR_SECRET_KEY,
                engine,
            )
            if not transcript:
                return jsonify({
                    "error": f"没有识别到清晰的{language_label}，请靠近麦克风再说一次。"
                }), 422
            return jsonify({"text": transcript, "language": language})

    except subprocess.TimeoutExpired:
        return jsonify({"error": "录音转换超时，请缩短录音后重试。"}), 504
    except TencentASRError as exc:
        print("腾讯云 ASR 调用失败:", exc)
        return jsonify({"error": str(exc)}), 502
    except Exception as exc:
        print("语音识别处理失败:", exc)
        return jsonify({"error": "语音识别处理失败，请稍后重试。"}), 500


@app.route('/test/start', methods=['POST'])
@login_required
def start_test():
    if g.current_user["account_type"] not in {"elder", "admin"}:
        abort(403)

    previous_id = session.get("test_id")
    if previous_id:
        TEST_SESSIONS.pop(previous_id, None)

    data = request.get_json(silent=True) or {}
    input_language = str(data.get("input_language", "browser")).strip().lower()
    if input_language not in {"browser", "cantonese", "chaoshan"}:
        input_language = "browser"

    test_id = uuid.uuid4().hex
    TEST_SESSIONS[test_id] = {
        "test_id": test_id,
        "user_id": g.current_user["id"],
        "question_index": 0,
        "score": 0,
        "domain_scores": {},
        "input_language": input_language,
        # 只记录题号和得分，不保存老人的原始回答。
        "scored_answers": [],
    }
    session["test_id"] = test_id

    reply = (
        "下面开始认知表现测试，共 10 题。请放松作答，可以打字，也可以点击说话。"
        "测试过程中不会立即提示对错。\n\n"
        "本测试仅用于项目演示和初步参考，不构成医学诊断。\n\n"
        + question_text(0)
    )
    return jsonify({
        "testing": True,
        "completed": False,
        "progress": f"1/{len(QUESTIONS)}",
        "reply": reply,
    })


@app.route('/test/answer', methods=['POST'])
@login_required
def answer_test():
    state = get_test_state()
    if state is None:
        return jsonify({"error": "当前没有正在进行的测试，请重新开始。"}), 409

    data = request.get_json(silent=True) or {}
    answer = str(data.get("message", "")).strip()
    if not answer:
        return jsonify({"error": "没有收到回答。"}), 400

    index = state["question_index"]
    question = QUESTIONS[index]
    earned = score_answer(question, answer)
    state["score"] += earned
    state["domain_scores"][question["domain"]] = (
        state["domain_scores"].get(question["domain"], 0) + earned
    )
    state["scored_answers"].append({
        "question_id": question["id"],
        "earned": earned,
        "maximum": question["points"],
    })
    state["question_index"] += 1

    if state["question_index"] >= len(QUESTIONS):
        maximum = total_points()
        percentage = round(state["score"] / maximum * 100)
        level = result_level(percentage)
        save_assessment(
            test_session_id=state["test_id"],
            user_id=state["user_id"],
            score=state["score"],
            maximum_score=maximum,
            percentage=percentage,
            level=level,
            domain_scores=state["domain_scores"],
            input_language=state["input_language"],
        )
        reply = format_result(state)
        test_id = session.pop("test_id", None)
        if test_id:
            TEST_SESSIONS.pop(test_id, None)
        return jsonify({
            "testing": False,
            "completed": True,
            "progress": f"{len(QUESTIONS)}/{len(QUESTIONS)}",
            "reply": reply,
        })

    next_index = state["question_index"]
    return jsonify({
        "testing": True,
        "completed": False,
        "progress": f"{next_index + 1}/{len(QUESTIONS)}",
        "reply": "回答已记录。\n\n" + question_text(next_index),
    })


@app.route('/test/status', methods=['GET'])
@login_required
def test_status():
    state = get_test_state()
    if state is None:
        return jsonify({"testing": False})

    index = state["question_index"]
    return jsonify({
        "testing": True,
        "progress": f"{index + 1}/{len(QUESTIONS)}",
        "reply": "检测到尚未完成的测试，我们从当前题继续。\n\n" + question_text(index),
    })


@app.route('/test/reset', methods=['POST'])
@login_required
def reset_test():
    test_id = session.pop("test_id", None)
    if test_id:
        TEST_SESSIONS.pop(test_id, None)
    return jsonify({"testing": False, "reply": "本次测试已退出，可以继续普通聊天。"})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
