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
import base64
import binascii
import csv
import io
import os
from pathlib import Path
import hashlib
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

from answer_processing import process_spoken_answer
from judgment_scoring import analyze_judgment_answer
from semantic_scoring import DEFAULT_MODEL_NAME, score_semantic_answer
import database
from database import (
    bind_elder,
    can_message_contact,
    count_idioms,
    count_unread_messages,
    count_assessments,
    create_drawing_submission,
    create_message,
    create_user,
    get_assessment,
    get_drawing_submission,
    get_knowledge_article,
    get_user_by_id,
    get_user_by_username,
    is_elder_bound_to,
    list_assessments,
    list_drawing_reviews,
    list_drawing_submissions_for_review,
    list_training_drawing_rows,
    list_user_drawing_submissions,
    list_bound_elders,
    list_elder_caregivers,
    list_message_contacts,
    list_messages,
    list_admin_knowledge_articles,
    list_knowledge_articles,
    list_knowledge_categories,
    list_recommended_knowledge,
    list_user_knowledge_submissions,
    list_users_with_stats,
    mark_messages_read,
    delete_user_drawing_submission,
    drawing_review_summary,
    save_assessment,
    save_drawing_review,
    set_knowledge_favorite,
    create_knowledge_article,
    create_knowledge_document,
    delete_knowledge_document,
    get_knowledge_document,
    knowledge_document_hash_exists,
    list_knowledge_documents,
    review_knowledge_article,
    submit_knowledge_article,
    unbind_elder,
    update_knowledge_article,
    update_knowledge_document_status,
    update_last_login,
    update_user_active,
    update_user_password,
)
from questions import (
    QUESTION_BANK_VERSION,
    TEST_QUESTION_COUNT,
    build_test_questions,
    domain_max_points,
    result_level,
    score_answer,
    total_points,
)
from tencent_asr import TencentASRError, recognize_pcm
from reports import build_assessment_report
from speech_analysis import analyze_pcm_file, build_speech_summary
from idiom_game import hints_for, new_game_state, play_turn
from drawing_scoring import (
    DrawingScoringError,
    TASKS as DRAWING_TASKS,
    extract_features as extract_drawing_features,
    normalize_strokes,
    score_drawing_assist,
)
from clock_model_service import (
    DEFAULT_CHECKPOINT as DEFAULT_CLOCK_MODEL_CHECKPOINT,
    score_clock_image,
)
from rag import (
    generate_knowledge_answer,
    is_casual_conversation,
    sync_knowledge_chunks,
)
from research_showcase import load_research_showcase
from i18n import normalise_language, translate, translate_payload

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
app.config["DRAWING_MODEL_DIR"] = os.getenv(
    "DRAWING_MODEL_DIR", str(Path(__file__).with_name("data") / "models")
)
app.config["DRAWING_CONFIDENCE_THRESHOLD"] = float(
    os.getenv("DRAWING_CONFIDENCE_THRESHOLD", "0.75")
)
app.config["CLOCK_DRAWING_MODEL_PATH"] = os.getenv(
    "CLOCK_DRAWING_MODEL_PATH"
) or str(DEFAULT_CLOCK_MODEL_CHECKPOINT)
app.config["CLOCK_DRAWING_CONFIDENCE_THRESHOLD"] = float(
    os.getenv("CLOCK_DRAWING_CONFIDENCE_THRESHOLD", "0.75")
)
app.config["SEMANTIC_SCORING_ENABLED"] = (
    os.getenv("SEMANTIC_SCORING_ENABLED", "true").lower() == "true"
)
app.config["SEMANTIC_MODEL_NAME"] = os.getenv(
    "SEMANTIC_MODEL_NAME", DEFAULT_MODEL_NAME
)
app.config["SEMANTIC_SIMILARITY_THRESHOLD"] = float(
    os.getenv("SEMANTIC_SIMILARITY_THRESHOLD", "0.55")
)
app.config["SEMANTIC_MARGIN_THRESHOLD"] = float(
    os.getenv("SEMANTIC_MARGIN_THRESHOLD", "0.06")
)
app.config["SEMANTIC_MODEL_LOCAL_ONLY"] = (
    os.getenv("SEMANTIC_MODEL_LOCAL_ONLY", "true").lower() == "true"
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
    # 普通话使用腾讯云作为主链路，产品页面在云端不可用时回退到
    # 浏览器 Web Speech API；离线数据集测试只评价腾讯云主链路。
    "mandarin": os.getenv(
        "TENCENT_ASR_ENGINE_MANDARIN",
        TENCENT_ASR_ENGINE,
    ),
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
    "mandarin": "普通话",
    "chaoshan": "潮汕话",
    "cantonese": "粤语",
}
FFMPEG_PATH = Path(__file__).with_name("ffmpeg.exe")

# ===========================================

database.init_app(app)

# 原型阶段将测试进度保存在服务进程内，避免把老人回答写入浏览器 Cookie。
TEST_SESSIONS = {}
IDIOM_GAME_SESSIONS = {}

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


def current_ui_language():
    """Return the interface language stored independently of speech mode."""
    return normalise_language(session.get("ui_language"))


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
        or request.path.startswith("/games/idiom/")
        or request.path.startswith("/drawing/api/")
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
    language = current_ui_language()
    return {
        "current_user": g.get("current_user"),
        "csrf_token": csrf_token,
        "current_language": language,
        "_": lambda text: translate(text, language),
        "unread_message_count": (
            count_unread_messages(g.current_user["id"])
            if g.get("current_user")
            and g.current_user["account_type"] in {"family", "community"}
            else 0
        ),
    }


@app.after_request
def localise_response(response):
    """Localise rendered HTML and the application's JSON status messages."""
    language = current_ui_language()
    if language != "en" or response.direct_passthrough or response.status_code == 204:
        return response
    if response.is_json:
        payload = response.get_json(silent=True)
        if payload is not None:
            response.set_data(json.dumps(
                translate_payload(payload, language),
                ensure_ascii=False,
            ))
    elif response.mimetype == "text/html":
        response.set_data(translate(response.get_data(as_text=True), language))
    return response


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


def community_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        if g.current_user["account_type"] != "community":
            abort(403)
        return view(*args, **kwargs)

    return wrapped_view


def elder_required(view):
    @wraps(view)
    @login_required
    def wrapped_view(*args, **kwargs):
        if g.current_user["account_type"] not in {"elder", "admin"}:
            if is_api_request():
                return jsonify({"error": "这个小游戏仅向老人端开放。"}), 403
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
        return translate("暂无", current_ui_language())
    parsed = datetime.fromisoformat(value)
    localized = parsed.astimezone(ZoneInfo("Asia/Shanghai"))
    if current_ui_language() == "en":
        return localized.strftime("%d %b %Y, %H:%M")
    return localized.strftime("%Y年%m月%d日 %H:%M")


@app.template_filter("language_name")
def language_name(value):
    label = {
        "browser": "普通话",
        "english": "English",
        "mandarin": "普通话",
        "cantonese": "粤语",
        "chaoshan": "潮汕话",
    }.get(value, value or "未记录")
    return translate(label, current_ui_language())


@app.template_filter("account_type_name")
def account_type_name(value):
    label = {
        "elder": "老人端",
        "family": "家属端",
        "community": "社区端",
        "admin": "后台管理员",
    }.get(value, value or "未设置")
    return translate(label, current_ui_language())


@app.template_filter("knowledge_audience_name")
def knowledge_audience_name(value):
    return {
        "all": "所有用户",
        "elder": "老人端",
        "family": "家属端",
        "community": "社区端",
    }.get(value, value or "未设置")


@app.template_filter("knowledge_status_name")
def knowledge_status_name(value):
    return {
        "draft": "草稿",
        "pending": "待审核",
        "published": "已发布",
        "rejected": "已退回",
        "disabled": "已停用",
    }.get(value, value or "未设置")


@app.template_filter("drawing_task_name")
def drawing_task_name(value):
    task = DRAWING_TASKS.get(value)
    return task["label"] if task else (value or "未知任务")


@app.template_filter("machine_status_name")
def machine_status_name(value):
    return {
        "not_available": "尚无可用模型",
        "predicted": "已生成辅助预测",
        "needs_review": "低置信度，等待人工复核",
    }.get(value, value or "未评分")


def knowledge_form_data():
    fields = {
        "title": 100,
        "summary": 240,
        "content": 8000,
        "category": 30,
        "keywords": 200,
        "domains": 200,
        "source_name": 160,
        "source_url": 500,
    }
    data = {
        name: str(request.form.get(name) or "").strip()[:maximum]
        for name, maximum in fields.items()
    }
    data["audience"] = str(request.form.get("audience") or "all").strip()
    if data["audience"] not in {"all", "elder", "family", "community"}:
        raise ValueError("请选择正确的适用对象。")
    if not all(data[name] for name in ("title", "summary", "content", "category")):
        raise ValueError("标题、摘要、正文和分类不能为空。")
    if len(data["content"]) < 40:
        raise ValueError("正文内容过短，请补充到至少40个字。")
    if data["source_url"]:
        parsed = urlsplit(data["source_url"])
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("来源链接必须是完整的 http 或 https 地址。")
    return data


def can_read_knowledge_article(article):
    if article is None:
        return False
    if g.current_user["role"] == "admin":
        return True
    return (
        article["status"] == "published"
        and article["audience"] in {"all", g.current_user["account_type"]}
    )


def extract_uploaded_knowledge(file, extension):
    if extension in {".txt", ".md"}:
        raw = file.read()
        for encoding in ("utf-8-sig", "utf-8", "gb18030"):
            try:
                return raw.decode(encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("文本编码无法识别，请将文件另存为UTF-8后重试。")
    if extension == ".docx":
        try:
            import docx
        except ImportError as error:
            raise ValueError("服务器尚未安装 python-docx，暂时无法解析DOCX。") from error
        document = docx.Document(file.stream)
        return "\n\n".join(
            paragraph.text.strip()
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        )
    if extension == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:
            raise ValueError("服务器尚未安装 pypdf，暂时无法解析PDF。") from error
        reader = PdfReader(file.stream)
        if reader.is_encrypted:
            raise ValueError("暂不支持加密PDF。")
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                pages.append(f"第{index}页\n{text}")
        return "\n\n".join(pages)
    raise ValueError("只支持TXT、Markdown、DOCX和可提取文字的PDF。")


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


def public_question(questions, index):
    question = questions[index]
    return {
        "number": index + 1,
        "total": len(questions),
        "prompt": question["prompt"],
        "domain": question.get("domain_label", question["domain"]),
        "response_type": question.get("response_type", "text"),
    }


def question_text(state, index):
    question = public_question(state["questions"], index)
    if state.get("ui_language") == "en":
        return (
            f"Task {question['number']} of {question['total']} "
            f"({question['domain']}):\n{question['prompt']}"
        )
    return f"第 {question['number']}/{question['total']} 题（{question['domain']}）：\n{question['prompt']}"


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


def attach_speech_observation(observation):
    """把服务端生成的特征绑定到当前题，原始音频不会进入测试状态。"""
    state = get_test_state()
    if state is None or not isinstance(observation, dict):
        return
    index = state["question_index"]
    if index >= len(state["questions"]):
        return
    safe_observation = {
        key: observation.get(key)
        for key in (
            "analysis_version",
            "valid",
            "quality",
            "audio_duration_ms",
            "speech_duration_ms",
            "silence_duration_ms",
            "initial_silence_ms",
            "pause_count",
            "pause_total_ms",
            "pause_mean_ms",
            "pause_max_ms",
            "voiced_ratio",
            "transcript_char_count",
            "speaking_rate_cpm",
        )
    }
    safe_observation["question_id"] = state["questions"][index]["id"]
    safe_observation["language"] = state["input_language"]
    state["pending_speech_observation"] = safe_observation


def get_idiom_game_state():
    game_id = session.get("idiom_game_id")
    if not game_id:
        return None
    state = IDIOM_GAME_SESSIONS.get(game_id)
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
    language = state.get("ui_language", "zh")
    domain_lines = [
        f"{translate(domain, language)}: {state['domain_scores'].get(domain, 0)}/{points}"
        for domain, points in domain_maximums.items()
    ]
    if language == "en":
        return (
            "Assessment complete.\n\n"
            f"Total score: {state['score']}/{maximum} ({percentage} points)\n\n"
            + "\n".join(domain_lines)
            + f"\n\nPerformance summary: {translate(result_level(percentage), language)}.\n\n"
            "Note: This result reflects this session only and is not a diagnosis of cognitive impairment or any other condition. "
            "Hearing, mood, fatigue, education, and speech-recognition accuracy may affect the result."
        )
    return (
        "测试完成。\n\n"
        f"总分：{state['score']}/{maximum}（{percentage}分）\n\n"
        + "\n".join(domain_lines)
        + f"\n\n表现反馈：{result_level(percentage)}。\n\n"
        "提示：本结果仅反映本次任务表现，不构成认知障碍或其他疾病诊断。"
        "听力、情绪、疲劳、教育程度和语音识别准确率都可能影响结果。"
    )


def finish_or_continue_test(state, acknowledgement="回答已记录。"):
    """在一次回答入账后，统一生成下一题或最终结果。"""
    question_count = len(state["questions"])
    if state["question_index"] >= question_count:
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
            question_bank_version=QUESTION_BANK_VERSION,
            question_ids=[question["id"] for question in state["questions"]],
            speech_metrics=build_speech_summary(
                state.get("speech_observations", [])
            ),
            scoring_audit={
                "scored_answers": state.get("scored_answers", []),
                "judgment_attempts": state.get("judgment_attempts", []),
            },
        )
        reply = format_result(state)
        test_id = session.pop("test_id", None)
        if test_id:
            TEST_SESSIONS.pop(test_id, None)
        return {
            "testing": False,
            "completed": True,
            "progress": f"{question_count}/{question_count}",
            "reply": reply,
            "question": None,
        }

    next_index = state["question_index"]
    return {
        "testing": True,
        "completed": False,
        "progress": f"{next_index + 1}/{question_count}",
        "reply": acknowledgement + "\n\n" + question_text(state, next_index),
        "question": public_question(state["questions"], next_index),
    }


@app.route("/")
def home():
    if g.current_user is None:
        return redirect(url_for("login"))
    return redirect(url_for("dashboard"))


@app.route("/language/<language>", methods=["POST"])
def set_language(language):
    session["ui_language"] = normalise_language(language)
    target = request.form.get("next") or request.referrer
    if target:
        parsed = urlsplit(target)
        if not parsed.netloc or parsed.netloc == request.host:
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            return redirect(path)
    return redirect(url_for("dashboard" if g.current_user else "login"))


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
            ui_language = current_ui_language()
            session.clear()
            session["ui_language"] = ui_language
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
    game_id = session.get("idiom_game_id")
    if game_id:
        IDIOM_GAME_SESSIONS.pop(game_id, None)
    ui_language = current_ui_language()
    session.clear()
    session["ui_language"] = ui_language
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


@app.route("/research/datasets")
@login_required
def research_datasets():
    """展示可公开的汇总指标，不暴露原始录音和逐条转写。"""
    return render_template(
        "research_datasets.html",
        showcase=load_research_showcase(),
    )


@app.route("/knowledge")
@login_required
def knowledge_center():
    query = str(request.args.get("q") or "").strip()[:80]
    category = str(request.args.get("category") or "").strip()[:30]
    favorites_only = request.args.get("favorites") == "1"
    return render_template(
        "knowledge_index.html",
        articles=list_knowledge_articles(
            g.current_user["account_type"],
            g.current_user["id"],
            query=query,
            category=category,
            favorites_only=favorites_only,
        ),
        categories=list_knowledge_categories(g.current_user["account_type"]),
        query=query,
        selected_category=category,
        favorites_only=favorites_only,
    )


@app.route("/knowledge/<int:article_id>")
@login_required
def knowledge_detail(article_id):
    article = get_knowledge_article(article_id, g.current_user["id"])
    if article is None:
        abort(404)
    if not can_read_knowledge_article(article):
        abort(403)
    return render_template("knowledge_detail.html", article=article)


@app.route("/knowledge/<int:article_id>/favorite", methods=["POST"])
@login_required
def knowledge_favorite(article_id):
    article = get_knowledge_article(article_id, g.current_user["id"])
    if article is None:
        abort(404)
    if not can_read_knowledge_article(article):
        abort(403)
    favorite = request.form.get("favorite") == "1"
    set_knowledge_favorite(g.current_user["id"], article_id, favorite)
    flash("已加入收藏。" if favorite else "已取消收藏。", "success")
    next_url = request.form.get("next")
    return redirect(
        next_url if is_safe_next_url(next_url) else
        url_for("knowledge_detail", article_id=article_id)
    )


@app.route("/knowledge/submissions")
@community_required
def knowledge_submissions():
    return render_template(
        "knowledge_submissions.html",
        articles=list_user_knowledge_submissions(g.current_user["id"]),
    )


@app.route("/knowledge/submissions/new", methods=["GET", "POST"])
@community_required
def new_knowledge_submission():
    if request.method == "POST":
        try:
            data = knowledge_form_data()
        except ValueError as error:
            flash(str(error), "error")
            return render_template(
                "knowledge_form.html",
                article=request.form,
                form_title="撰写社区知识草稿",
                form_action=url_for("new_knowledge_submission"),
                admin_mode=False,
            )
        article_id = create_knowledge_article(
            data,
            g.current_user["id"],
            status="draft",
        )
        flash("草稿已保存。确认内容和来源后，请提交管理员审核。", "success")
        return redirect(url_for("edit_knowledge_submission", article_id=article_id))
    return render_template(
        "knowledge_form.html",
        article=None,
        form_title="撰写社区知识草稿",
        form_action=url_for("new_knowledge_submission"),
        admin_mode=False,
    )


@app.route(
    "/knowledge/submissions/<int:article_id>/edit",
    methods=["GET", "POST"],
)
@community_required
def edit_knowledge_submission(article_id):
    article = get_knowledge_article(article_id, g.current_user["id"])
    if article is None:
        abort(404)
    if (
        article["author_id"] != g.current_user["id"]
        or article["status"] not in {"draft", "rejected"}
    ):
        abort(403)
    if request.method == "POST":
        try:
            data = knowledge_form_data()
        except ValueError as error:
            flash(str(error), "error")
            return render_template(
                "knowledge_form.html",
                article=request.form,
                form_title="编辑社区知识草稿",
                form_action=url_for(
                    "edit_knowledge_submission",
                    article_id=article_id,
                ),
                admin_mode=False,
            )
        update_knowledge_article(article_id, data, status="draft")
        flash("草稿已保存。", "success")
        return redirect(url_for("edit_knowledge_submission", article_id=article_id))
    return render_template(
        "knowledge_form.html",
        article=article,
        form_title="编辑社区知识草稿",
        form_action=url_for("edit_knowledge_submission", article_id=article_id),
        admin_mode=False,
    )


@app.route(
    "/knowledge/submissions/<int:article_id>/submit",
    methods=["POST"],
)
@community_required
def send_knowledge_submission(article_id):
    if not submit_knowledge_article(article_id, g.current_user["id"]):
        abort(403)
    flash("已提交审核。管理员发布前，文章不会出现在知识中心。", "success")
    return redirect(url_for("knowledge_submissions"))


@app.route("/admin/knowledge")
@admin_required
def admin_knowledge():
    selected_status = str(request.args.get("status") or "").strip()
    if selected_status not in {
        "", "draft", "pending", "published", "rejected", "disabled"
    }:
        selected_status = ""
    return render_template(
        "admin_knowledge.html",
        articles=list_admin_knowledge_articles(selected_status),
        selected_status=selected_status,
    )


@app.route("/admin/knowledge/new", methods=["GET", "POST"])
@admin_required
def admin_new_knowledge():
    if request.method == "POST":
        try:
            data = knowledge_form_data()
        except ValueError as error:
            flash(str(error), "error")
            return render_template(
                "knowledge_form.html",
                article=request.form,
                form_title="新建知识文章",
                form_action=url_for("admin_new_knowledge"),
                admin_mode=True,
            )
        status = request.form.get("status")
        if status not in {"draft", "published"}:
            status = "draft"
        create_knowledge_article(data, g.current_user["id"], status=status)
        flash("知识文章已保存。", "success")
        return redirect(url_for("admin_knowledge"))
    return render_template(
        "knowledge_form.html",
        article=None,
        form_title="新建知识文章",
        form_action=url_for("admin_new_knowledge"),
        admin_mode=True,
    )


@app.route(
    "/admin/knowledge/<int:article_id>/edit",
    methods=["GET", "POST"],
)
@admin_required
def admin_edit_knowledge(article_id):
    article = get_knowledge_article(article_id, g.current_user["id"])
    if article is None:
        abort(404)
    if request.method == "POST":
        try:
            data = knowledge_form_data()
        except ValueError as error:
            flash(str(error), "error")
            return render_template(
                "knowledge_form.html",
                article=request.form,
                form_title="编辑知识文章",
                form_action=url_for(
                    "admin_edit_knowledge",
                    article_id=article_id,
                ),
                admin_mode=True,
            )
        status = request.form.get("status")
        if status not in {"draft", "published", "disabled"}:
            status = article["status"]
        update_knowledge_article(article_id, data, status=status)
        flash("知识文章已更新。", "success")
        return redirect(url_for("admin_knowledge"))
    return render_template(
        "knowledge_form.html",
        article=article,
        form_title="编辑知识文章",
        form_action=url_for("admin_edit_knowledge", article_id=article_id),
        admin_mode=True,
    )


@app.route(
    "/admin/knowledge/<int:article_id>/review",
    methods=["GET", "POST"],
)
@admin_required
def admin_review_knowledge(article_id):
    article = get_knowledge_article(article_id, g.current_user["id"])
    if article is None:
        abort(404)
    if request.method == "POST":
        decision = request.form.get("decision")
        if decision not in {"published", "rejected"}:
            flash("请选择“通过发布”或“退回修改”。", "error")
        elif not review_knowledge_article(
            article_id,
            g.current_user["id"],
            decision,
            str(request.form.get("review_notes") or "").strip()[:1000],
        ):
            flash("该文章当前不在待审核状态。", "error")
        else:
            flash(
                "文章已审核并发布。" if decision == "published"
                else "文章已退回社区作者修改。",
                "success",
            )
            return redirect(url_for("admin_knowledge"))
        article = get_knowledge_article(article_id, g.current_user["id"])
    return render_template("knowledge_review.html", article=article)


@app.route("/admin/rag-documents")
@admin_required
def admin_rag_documents():
    selected_status = str(request.args.get("status") or "").strip()
    if selected_status not in {"", "draft", "published", "disabled"}:
        selected_status = ""
    return render_template(
        "admin_rag_documents.html",
        documents=list_knowledge_documents(selected_status),
        selected_status=selected_status,
    )


@app.route("/admin/rag-documents/upload", methods=["POST"])
@admin_required
def admin_upload_rag_document():
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        flash("请选择需要上传的知识文档。", "error")
        return redirect(url_for("admin_rag_documents"))

    filename = str(uploaded.filename).replace("\\", "/").split("/")[-1].strip()[:200]
    extension = Path(filename).suffix.lower()
    if extension not in {".txt", ".md", ".docx", ".pdf"}:
        flash("只支持TXT、Markdown、DOCX和可提取文字的PDF。", "error")
        return redirect(url_for("admin_rag_documents"))

    audience = str(request.form.get("audience") or "all").strip()
    if audience not in {"all", "elder", "family", "community"}:
        flash("请选择正确的适用对象。", "error")
        return redirect(url_for("admin_rag_documents"))
    status = str(request.form.get("status") or "draft").strip()
    if status not in {"draft", "published"}:
        status = "draft"
    source_url = str(request.form.get("source_url") or "").strip()[:500]
    if source_url:
        parsed = urlsplit(source_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            flash("来源链接必须是完整的HTTP或HTTPS地址。", "error")
            return redirect(url_for("admin_rag_documents"))

    try:
        content = extract_uploaded_knowledge(uploaded, extension).strip()
        if len(content) < 40:
            raise ValueError("文档可提取正文过短，请检查文件内容。")
        if len(content) > 500_000:
            raise ValueError("文档正文超过50万字，请拆分后上传。")
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if knowledge_document_hash_exists(digest):
            raise ValueError("相同内容的文档已经存在，请不要重复上传。")

        title = str(request.form.get("title") or Path(filename).stem).strip()[:120]
        if not title:
            raise ValueError("请填写文档标题。")
        create_knowledge_document(
            {
                "document_id": uuid.uuid4().hex,
                "title": title,
                "filename": filename,
                "content": content,
                "category": str(
                    request.form.get("category") or "健康教育"
                ).strip()[:40],
                "audience": audience,
                "source_name": str(
                    request.form.get("source_name") or ""
                ).strip()[:160],
                "source_url": source_url,
                "content_hash": digest,
                "status": status,
            },
            g.current_user["id"],
        )
        sync_knowledge_chunks()
        flash(
            "文档已发布并进入知识检索。"
            if status == "published"
            else "文档草稿已保存，发布前不会用于回答。",
            "success",
        )
    except ValueError as error:
        flash(str(error), "error")
    except Exception:
        app.logger.exception("上传RAG知识文档失败")
        flash("文档处理失败，请检查格式后重试。", "error")
    return redirect(url_for("admin_rag_documents"))


@app.route(
    "/admin/rag-documents/<document_id>/status",
    methods=["POST"],
)
@admin_required
def admin_set_rag_document_status(document_id):
    document = get_knowledge_document(document_id)
    if document is None:
        abort(404)
    status = str(request.form.get("status") or "").strip()
    if status not in {"draft", "published", "disabled"}:
        flash("不支持的文档状态。", "error")
        return redirect(url_for("admin_rag_documents"))
    update_knowledge_document_status(
        document_id,
        status,
        g.current_user["id"],
        str(request.form.get("review_notes") or "").strip()[:1000],
    )
    sync_knowledge_chunks()
    flash(
        "文档已发布并可用于知识问答。"
        if status == "published"
        else "文档已停止用于知识问答。",
        "success",
    )
    return redirect(url_for("admin_rag_documents"))


@app.route(
    "/admin/rag-documents/<document_id>/delete",
    methods=["POST"],
)
@admin_required
def admin_delete_rag_document(document_id):
    if not delete_knowledge_document(document_id):
        abort(404)
    sync_knowledge_chunks()
    flash("知识文档及其检索片段已删除。", "success")
    return redirect(url_for("admin_rag_documents"))


@app.route("/admin/rag-documents/reindex", methods=["POST"])
@admin_required
def admin_reindex_rag_documents():
    sync_knowledge_chunks()
    flash("知识文章和已发布文档的检索索引已重新构建。", "success")
    return redirect(url_for("admin_rag_documents"))


@app.route("/assistant")
@login_required
def assistant_page():
    if g.current_user["account_type"] in {"family", "community"}:
        abort(403)
    return render_template("index.html")


@app.route("/games/idiom")
@elder_required
def idiom_game_page():
    return render_template(
        "idiom_game.html",
        idiom_count=count_idioms(),
    )


@app.route("/games/idiom/start", methods=["POST"])
@elder_required
def start_idiom_game():
    previous_id = session.get("idiom_game_id")
    if previous_id:
        IDIOM_GAME_SESSIONS.pop(previous_id, None)

    game_id = uuid.uuid4().hex
    state = new_game_state(g.current_user["id"])
    IDIOM_GAME_SESSIONS[game_id] = state
    session["idiom_game_id"] = game_id
    return jsonify({
        "active": True,
        "computer_idiom": state["computer_idiom"],
        "required_character": state["required_character"],
        "rounds": 0,
        "hints": hints_for(state),
    })


@app.route("/games/idiom/play", methods=["POST"])
@elder_required
def submit_idiom():
    state = get_idiom_game_state()
    if state is None:
        return jsonify({"error": "游戏尚未开始，请先点击“开始游戏”。"}), 409

    data = request.get_json(silent=True) or {}
    result = play_turn(state, data.get("idiom"))
    if not result["ok"]:
        return jsonify(result), 422
    if result.get("completed"):
        game_id = session.pop("idiom_game_id", None)
        if game_id:
            IDIOM_GAME_SESSIONS.pop(game_id, None)
    return jsonify(result)


@app.route("/games/idiom/reset", methods=["POST"])
@elder_required
def reset_idiom_game():
    game_id = session.pop("idiom_game_id", None)
    if game_id:
        IDIOM_GAME_SESSIONS.pop(game_id, None)
    return jsonify({"active": False})


DRAWING_RUBRICS = {
    "trail": [
        ("sequence", "连接顺序完整且符合任务要求"),
        ("no_uncorrected_error", "没有未自行纠正的错误"),
    ],
    "cube": [
        ("three_dimensional", "呈现三维结构"),
        ("all_lines", "必要线条完整"),
        ("no_extra_lines", "没有明显多余线条"),
        ("parallel_edges", "对应边大致平行且长度相近"),
    ],
    "clock": [
        ("contour", "轮廓符合评分要求"),
        ("numbers", "数字完整、顺序和位置符合要求"),
        ("hands", "指针数量、方向和中心符合要求"),
    ],
}


def decode_drawing_png(data_url):
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise DrawingScoringError("绘图预览必须是PNG格式。")
    try:
        content = base64.b64decode(data_url[len(prefix):], validate=True)
    except (binascii.Error, ValueError) as error:
        raise DrawingScoringError("绘图预览无法解析。") from error
    if not content.startswith(b"\x89PNG\r\n\x1a\n"):
        raise DrawingScoringError("绘图预览不是有效PNG文件。")
    if len(content) > 3 * 1024 * 1024:
        raise DrawingScoringError("绘图文件过大，请清空后重新绘制。")
    return content


def drawing_machine_prediction(task_type, image_png, features):
    """钟表优先使用图像模型，其他任务保留原轨迹特征基线。"""
    if task_type == "clock":
        return score_clock_image(
            image_png,
            checkpoint_path=app.config["CLOCK_DRAWING_MODEL_PATH"],
            confidence_threshold=app.config[
                "CLOCK_DRAWING_CONFIDENCE_THRESHOLD"
            ],
        )
    return score_drawing_assist(
        features,
        task_type,
        app.config["DRAWING_MODEL_DIR"],
        app.config["DRAWING_CONFIDENCE_THRESHOLD"],
    )


@app.route("/drawing-lab")
@elder_required
def drawing_lab_page():
    return render_template(
        "drawing_lab.html",
        drawing_tasks=DRAWING_TASKS,
        submissions=list_user_drawing_submissions(g.current_user["id"]),
    )


@app.route("/drawing/api/submit", methods=["POST"])
@elder_required
def submit_drawing():
    data = request.get_json(silent=True) or {}
    task_type = str(data.get("task_type") or "").strip()
    if task_type not in DRAWING_TASKS:
        return jsonify({"error": "请选择正确的绘图任务。"}), 422
    if data.get("research_consent") is not True:
        return jsonify({"error": "请先阅读并确认研究数据说明。"}), 422
    try:
        width = int(data.get("canvas_width"))
        height = int(data.get("canvas_height"))
        strokes = normalize_strokes(data.get("strokes"), width, height)
        features = extract_drawing_features(strokes, width, height)
        image_png = decode_drawing_png(data.get("image_png"))
        prediction = drawing_machine_prediction(
            task_type, image_png, features
        )
    except (DrawingScoringError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 422

    database_id = create_drawing_submission(
        submission_id=uuid.uuid4().hex,
        user_id=g.current_user["id"],
        task_type=task_type,
        source_scale="research-drawing-v1",
        canvas_width=width,
        canvas_height=height,
        strokes=strokes,
        image_png=image_png,
        features=features,
        machine_prediction=prediction,
        research_consent=True,
    )
    return jsonify(
        {
            "submission_id": database_id,
            "machine_score": prediction.get("score"),
            "machine_confidence": prediction.get("confidence"),
            "machine_status": prediction["status"],
            "score_label": prediction.get("score_label"),
            "source_label_system": prediction.get("source_label_system"),
            "message": (
                "绘图已保存，并生成Shulman 0～5实验预测；这不是MoCA分数，仍需人工复核。"
                if prediction.get("score") is not None
                else "绘图已保存，模型未给出可靠结果，当前需要人工复核。"
            ),
        }
    ), 201


@app.route("/drawing-lab/<int:submission_id>/delete", methods=["POST"])
@elder_required
def delete_drawing(submission_id):
    if not delete_user_drawing_submission(submission_id, g.current_user["id"]):
        abort(404)
    flash("绘图样本及其复核记录已删除。", "success")
    return redirect(url_for("drawing_lab_page"))


@app.route("/admin/drawings")
@admin_required
def admin_drawings():
    return render_template(
        "admin_drawings.html",
        submissions=list_drawing_submissions_for_review(),
    )


@app.route("/admin/drawings/<int:submission_id>/image")
@admin_required
def admin_drawing_image(submission_id):
    submission = get_drawing_submission(submission_id)
    if submission is None:
        abort(404)
    response = app.response_class(submission["image_png"], mimetype="image/png")
    response.headers["Cache-Control"] = "no-store, private"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


@app.route(
    "/admin/drawings/<int:submission_id>/review", methods=["GET", "POST"]
)
@admin_required
def admin_drawing_review(submission_id):
    submission = get_drawing_submission(submission_id)
    if submission is None:
        abort(404)
    task = DRAWING_TASKS[submission["task_type"]]
    summary = drawing_review_summary(submission_id)

    if request.method == "POST":
        review_role = str(request.form.get("review_role") or "rater")
        if review_role not in {"rater", "adjudicator"}:
            flash("复核角色不正确。", "error")
            return redirect(url_for("admin_drawing_review", submission_id=submission_id))
        try:
            score = int(request.form.get("score"))
        except (TypeError, ValueError):
            score = -1
        if not 0 <= score <= task["maximum_score"]:
            flash(f"本任务得分必须在0到{task['maximum_score']}之间。", "error")
            return redirect(url_for("admin_drawing_review", submission_id=submission_id))

        existing = summary["reviews"]
        rater_ids = {
            row["reviewer_id"] for row in existing if row["review_role"] == "rater"
        }
        if review_role == "rater" and len(rater_ids) >= 2 and g.current_user["id"] not in rater_ids:
            flash("该样本已经由两名独立评分者完成初评。", "error")
            return redirect(url_for("admin_drawing_review", submission_id=submission_id))
        if review_role == "adjudicator":
            if len(rater_ids) < 2:
                flash("必须先完成两名独立评分者的初评，才能仲裁。", "error")
                return redirect(url_for("admin_drawing_review", submission_id=submission_id))
            if g.current_user["id"] in rater_ids:
                flash("仲裁员必须与两名初评人员不同。", "error")
                return redirect(url_for("admin_drawing_review", submission_id=submission_id))

        rubric = {
            key: request.form.get(f"criterion_{key}") == "on"
            for key, _label in DRAWING_RUBRICS[submission["task_type"]]
        }
        save_drawing_review(
            submission_id=submission_id,
            reviewer_id=g.current_user["id"],
            review_role=review_role,
            score=score,
            rubric=rubric,
            notes=str(request.form.get("notes") or "").strip()[:1000],
        )
        flash("复核结果已保存。", "success")
        return redirect(url_for("admin_drawing_review", submission_id=submission_id))

    return render_template(
        "drawing_review.html",
        submission=submission,
        task=task,
        rubric_items=DRAWING_RUBRICS[submission["task_type"]],
        summary=summary,
    )


@app.route("/admin/drawings/dataset.csv")
@admin_required
def export_drawing_dataset():
    rows = list_training_drawing_rows()
    output = io.StringIO(newline="")
    feature_names = [
        "stroke_count", "point_count", "duration_s", "ink_length_norm",
        "bbox_width_ratio", "bbox_height_ratio", "coverage_ratio",
        "closed_stroke_ratio", "pen_lift_rate",
    ]
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "submission_id", "task_type", "score", "label_source",
            "feature_version", *feature_names,
        ],
    )
    writer.writeheader()
    for row in rows:
        features = row["features"]
        writer.writerow(
            {
                "submission_id": row["submission_id"],
                "task_type": row["task_type"],
                "score": row["score"],
                "label_source": row["label_source"],
                "feature_version": features.get("feature_version"),
                **{name: features.get(name) for name in feature_names},
            }
        )
    response = app.response_class(output.getvalue(), mimetype="text/csv")
    response.headers["Content-Disposition"] = (
        "attachment; filename=drawing_training_dataset.csv"
    )
    response.headers["Cache-Control"] = "no-store, private"
    return response


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


def parse_speech_metrics(assessment):
    raw_metrics = assessment["speech_metrics"]
    if not raw_metrics:
        return None
    try:
        metrics = json.loads(raw_metrics)
    except (TypeError, json.JSONDecodeError):
        return None
    return metrics if isinstance(metrics, dict) else None


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
        speech_metrics=parse_speech_metrics(assessment),
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
    language = current_ui_language()
    report = translate_payload(
        build_assessment_report(assessment, scores, maximums),
        language,
    )
    return render_template(
        "assessment_report.html",
        assessment=assessment,
        domain_scores=scores,
        domain_maximums=maximums,
        speech_metrics=parse_speech_metrics(assessment),
        report=report,
        recommended_articles=(
            [] if language == "en" else list_recommended_knowledge(
                g.current_user["account_type"],
                g.current_user["id"],
                [item["domain"] for item in report["priorities"]],
            )
        ),
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


@app.route('/chat', methods=['POST'])
@login_required
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = str(payload.get('message') or '').strip()
    use_knowledge = bool(payload.get('use_knowledge', True))

    if not user_input:
        return jsonify({'error': '没有收到消息'}), 400
    if len(user_input) > 2000:
        return jsonify({'error': '单次提问不能超过2000个字。'}), 400

    try:
        if use_knowledge and not is_casual_conversation(user_input):
            knowledge_args = (
                user_input,
                g.current_user["account_type"],
                deepseek_client,
                DEEPSEEK_MODEL,
            )
            if current_ui_language() == "en":
                ai_reply, references, urgent = generate_knowledge_answer(
                    *knowledge_args,
                    language="en",
                )
            else:
                # Keep the original four-argument path stable for tests and
                # existing integrations that mock the knowledge function.
                ai_reply, references, urgent = generate_knowledge_answer(
                    *knowledge_args,
                )
            return jsonify({
                'reply': ai_reply,
                'references': references,
                'knowledge_used': bool(references),
                'urgent': urgent,
            })

        ai_reply = get_ai_reply(
            user_input,
            (
                "You are the companion assistant for Memory Lantern. "
                "Reply in warm, brief, easy-to-understand English suitable for older adults. "
                "Never claim to provide a medical diagnosis."
                if current_ui_language() == "en" else
                "你是“忆路明灯”的智慧陪伴助手。请使用温和、简短、容易理解的中文回应，"
                "适合与老年人交流。不要声称自己能够进行医疗诊断。"
            ),
        )
        return jsonify({
            'reply': ai_reply,
            'references': [],
            'knowledge_used': False,
            'urgent': False,
            'conversation_mode': (
                'casual'
                if use_knowledge and is_casual_conversation(user_input)
                else 'companion'
            ),
        })

    except Exception:
        app.logger.exception("聊天请求处理失败")
        return jsonify({
            'error': "服务暂时无法回答，请稍后再试。",
            'references': [],
        }), 500


@app.route('/speech/recognize', methods=['POST'])
@login_required
def recognize_speech():
    language = str(request.form.get("language", "chaoshan")).strip().lower()
    engine = TENCENT_ASR_ENGINES.get(language)
    if engine is None:
        return jsonify({"error": "不支持所选择的语音类型。"}), 400
    language_label = SPEECH_LANGUAGE_LABELS[language]

    if not all([
        TENCENT_ASR_APP_ID,
        TENCENT_ASR_SECRET_ID,
        TENCENT_ASR_SECRET_KEY,
    ]):
        return jsonify({
            "error": "腾讯云语音识别尚未配置，请填写 .env 中的腾讯云密钥。",
            "fallback_recommended": language == "mandarin",
            "error_kind": "configuration",
        }), 503

    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"error": "没有收到录音文件。"}), 400

    if not FFMPEG_PATH.exists():
        return jsonify({"error": "项目目录中缺少 ffmpeg.exe。"}), 500

    try:
        with tempfile.TemporaryDirectory(prefix="memory_lantern_asr_") as temp_dir:
            input_path = Path(temp_dir) / "recording.webm"
            pcm_path = Path(temp_dir) / "recording.pcm"
            audio_file.save(input_path)

            conversion = convert_audio_to_pcm(input_path, pcm_path)
            if conversion.returncode != 0 or not pcm_path.exists():
                app.logger.warning("音频转码失败: %s", conversion.stderr)
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
                    "error": f"没有识别到清晰的{language_label}，请靠近麦克风再说一次。",
                    "fallback_recommended": language == "mandarin",
                    "error_kind": "empty_transcript",
                }), 422
            observation = analyze_pcm_file(pcm_path, transcript=transcript)
            attach_speech_observation(observation)
            return jsonify({
                "text": transcript,
                "language": language,
                "speech_observation": observation,
            })

    except subprocess.TimeoutExpired:
        return jsonify({"error": "录音转换超时，请缩短录音后重试。"}), 504
    except TencentASRError as exc:
        print("腾讯云 ASR 调用失败:", exc)
        return jsonify({
            "error": str(exc),
            "fallback_recommended": language == "mandarin",
            "error_kind": "cloud_unavailable",
        }), 502
    except Exception as exc:
        print("语音识别处理失败:", exc)
        return jsonify({
            "error": "语音识别处理失败，请稍后重试。",
            "fallback_recommended": language == "mandarin",
            "error_kind": "internal_error",
        }), 500


def convert_audio_to_pcm(input_path, pcm_path):
    return subprocess.run(
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


@app.route('/speech/analyze', methods=['POST'])
@login_required
def analyze_speech():
    """分析普通话浏览器录音；只返回并保存特征，不保留音频。"""
    audio_file = request.files.get("audio")
    if audio_file is None or not audio_file.filename:
        return jsonify({"error": "没有收到录音文件。"}), 400
    if not FFMPEG_PATH.exists():
        return jsonify({"error": "项目目录中缺少 ffmpeg.exe。"}), 500

    transcript = str(request.form.get("transcript", "")).strip()
    try:
        with tempfile.TemporaryDirectory(prefix="memory_lantern_analysis_") as temp_dir:
            input_path = Path(temp_dir) / "recording.webm"
            pcm_path = Path(temp_dir) / "recording.pcm"
            audio_file.save(input_path)
            conversion = convert_audio_to_pcm(input_path, pcm_path)
            if conversion.returncode != 0 or not pcm_path.exists():
                app.logger.warning("语音行为分析转码失败: %s", conversion.stderr)
                return jsonify({"error": "录音格式转换失败。"}), 400

            observation = analyze_pcm_file(pcm_path, transcript=transcript)
            attach_speech_observation(observation)
            return jsonify({"speech_observation": observation})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "录音分析超时，请缩短录音后重试。"}), 504
    except Exception:
        app.logger.exception("语音行为分析失败")
        return jsonify({"error": "语音行为分析失败。"}), 500


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
    if input_language not in {"browser", "english", "mandarin", "cantonese", "chaoshan"}:
        input_language = "browser"

    test_id = uuid.uuid4().hex
    ui_language = current_ui_language()
    selected_questions = build_test_questions(language=ui_language)
    TEST_SESSIONS[test_id] = {
        "test_id": test_id,
        "user_id": g.current_user["id"],
        "question_index": 0,
        "questions": selected_questions,
        "score": 0,
        "domain_scores": {},
        "input_language": input_language,
        "ui_language": ui_language,
        # 只记录题号和得分，不保存老人的原始回答。
        "scored_answers": [],
        "speech_observations": [],
        "pending_speech_observation": None,
        "judgment_clarifications": {},
        "judgment_attempts": [],
    }
    session["test_id"] = test_id

    intro = (
        "The cognitive performance assessment now begins. It has 11 tasks: the first 10 are scored "
        "and the final clock-drawing task is an experimental observation. Please answer at your own "
        "pace by typing or speaking. Correctness is not shown during the assessment.\n\n"
        "This assessment is for project demonstration and preliminary reference only. "
        "It is not a medical diagnosis.\n\n"
        if ui_language == "en" else
        "下面开始认知表现测试，共11题：前10题为原有计分题，第11题为画钟辅助观察。"
        "请放松作答，可以打字，也可以点击说话。测试过程中不会立即提示对错。\n\n"
        "本测试仅用于项目演示和初步参考，不构成医学诊断。\n\n"
    )
    reply = intro + question_text(TEST_SESSIONS[test_id], 0)
    return jsonify({
        "testing": True,
        "completed": False,
        "progress": f"1/{TEST_QUESTION_COUNT}",
        "reply": reply,
        "question": public_question(selected_questions, 0),
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
    question = state["questions"][index]
    if question.get("response_type") == "drawing":
        return jsonify({"error": "本题需要在画布上完成并提交。"}), 422
    processed_answer = process_spoken_answer(
        answer,
        language=state.get("ui_language", "zh"),
    )
    judgment_analysis = None
    semantic_analysis = None
    resolved_judgment_decision = None
    judgment_scoring_source = None
    if question.get("type") == "judgment":
        judgment_analysis = analyze_judgment_answer(
            question["id"],
            processed_answer.final_answer,
            language=state.get("ui_language", "zh"),
        )
        resolved_judgment_decision = judgment_analysis.decision
        judgment_scoring_source = "rules"

        # 规则已经识别出安全、危险或矛盾行为时，以可解释规则为准；
        # 只有没有命中规则的自由表达才交给本地语义模型。
        if judgment_analysis.decision == "uncertain":
            semantic_analysis = score_semantic_answer(
                question["id"],
                processed_answer.final_answer,
                language=state.get("ui_language", "zh"),
                enabled=app.config["SEMANTIC_SCORING_ENABLED"],
                model_name=app.config["SEMANTIC_MODEL_NAME"],
                similarity_threshold=app.config["SEMANTIC_SIMILARITY_THRESHOLD"],
                margin_threshold=app.config["SEMANTIC_MARGIN_THRESHOLD"],
                local_files_only=app.config["SEMANTIC_MODEL_LOCAL_ONLY"],
            )
            if semantic_analysis.decision in {
                "correct", "incorrect", "dangerous", "partial"
            }:
                resolved_judgment_decision = semantic_analysis.decision
                judgment_scoring_source = "semantic_model"

        if resolved_judgment_decision in {"conflict", "uncertain", "partial"}:
            clarifications = state.setdefault("judgment_clarifications", {})
            clarification_count = int(clarifications.get(question["id"], 0))
            attempt_record = {
                "question_id": question["id"],
                "clarification_number": clarification_count + 1,
                "resolved_judgment_decision": resolved_judgment_decision,
                "judgment_scoring_source": judgment_scoring_source,
                **processed_answer.behavior_metrics(),
                **judgment_analysis.audit_record(),
            }
            if semantic_analysis is not None:
                attempt_record.update(semantic_analysis.audit_record())
            state.setdefault("judgment_attempts", []).append(attempt_record)

            pending_observation = state.pop("pending_speech_observation", None)
            if pending_observation is not None:
                pending_observation.update(processed_answer.behavior_metrics())
                pending_observation.update({
                    "judgment_decision": judgment_analysis.decision,
                    "judgment_rule_version": judgment_analysis.rule_version,
                    "resolved_judgment_decision": resolved_judgment_decision,
                    "judgment_scoring_source": judgment_scoring_source,
                })
                state["speech_observations"].append(pending_observation)
            state["pending_speech_observation"] = None

            if clarification_count < 1:
                clarifications[question["id"]] = clarification_count + 1
                reply = (
                    "Your answer included different or unclear actions. Please answer once more: "
                    "what would you do in this situation?"
                    if state.get("ui_language") == "en" else
                    "刚才的回答中包含不同做法，或者还不够明确。请您再明确说一次：遇到这种情况，您会怎么做？"
                )
                return jsonify({
                    "testing": True,
                    "completed": False,
                    "clarification_required": True,
                    "progress": f"{index + 1}/{len(state['questions'])}",
                    "reply": reply,
                    "question": public_question(state["questions"], index),
                })
        earned = question["points"] if resolved_judgment_decision == "correct" else 0
    else:
        earned = score_answer(question, processed_answer.final_answer)
    state["score"] += earned
    state["domain_scores"][question["domain"]] = (
        state["domain_scores"].get(question["domain"], 0) + earned
    )
    scored_record = {
        "question_id": question["id"],
        "earned": earned,
        "maximum": question["points"],
        **processed_answer.behavior_metrics(),
    }
    if judgment_analysis is not None:
        scored_record.update(judgment_analysis.audit_record())
        scored_record.update({
            "resolved_judgment_decision": resolved_judgment_decision,
            "judgment_scoring_source": judgment_scoring_source,
        })
    if semantic_analysis is not None:
        scored_record.update(semantic_analysis.audit_record())
    state["scored_answers"].append(scored_record)
    pending_observation = state.pop("pending_speech_observation", None)
    if pending_observation is not None:
        pending_observation.update(processed_answer.behavior_metrics())
        state["speech_observations"].append(pending_observation)
    state["pending_speech_observation"] = None
    state["question_index"] += 1
    return jsonify(finish_or_continue_test(state))


@app.route('/test/drawing-answer', methods=['POST'])
@login_required
def answer_test_drawing():
    state = get_test_state()
    if state is None:
        return jsonify({"error": "当前没有正在进行的测试，请重新开始。"}), 409

    index = state["question_index"]
    question = state["questions"][index]
    if question.get("response_type") != "drawing":
        return jsonify({"error": "当前题不是绘图题。"}), 409

    data = request.get_json(silent=True) or {}
    if data.get("research_consent") is not True:
        return jsonify({"error": "请先阅读并确认绘图数据说明。"}), 422
    try:
        width = int(data.get("canvas_width"))
        height = int(data.get("canvas_height"))
        strokes = normalize_strokes(data.get("strokes"), width, height)
        features = extract_drawing_features(strokes, width, height)
        image_png = decode_drawing_png(data.get("image_png"))
        prediction = drawing_machine_prediction("clock", image_png, features)
    except (DrawingScoringError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 422

    submission_id = create_drawing_submission(
        submission_id=uuid.uuid4().hex,
        user_id=g.current_user["id"],
        task_type="clock",
        source_scale="clock-observation-shulman-v1",
        canvas_width=width,
        canvas_height=height,
        strokes=strokes,
        image_png=image_png,
        features=features,
        machine_prediction=prediction,
        research_consent=True,
    )
    state["scored_answers"].append({
        "question_id": question["id"],
        "earned": 0,
        "maximum": 0,
        "drawing_submission_id": submission_id,
    })
    state["pending_speech_observation"] = None
    state["question_index"] += 1

    result = finish_or_continue_test(
        state,
        acknowledgement=(
            "The clock drawing has been saved. The machine output is an experimental Shulman "
            "0–5 prediction; it is excluded from the 14-point total and does not replace human MoCA scoring."
            if state.get("ui_language") == "en" else
            "画钟已保存。机器结果仅作为Shulman 0～5实验预测，不计入14分总分，也不替代人工MoCA评分。"
        ),
    )
    result["drawing_result"] = {
        "submission_id": submission_id,
        "score": prediction.get("score"),
        "score_label": prediction.get("score_label"),
        "confidence": prediction.get("confidence"),
        "status": prediction.get("status"),
        "source_label_system": prediction.get("source_label_system"),
    }
    return jsonify(result), 201


@app.route('/test/status', methods=['GET'])
@login_required
def test_status():
    state = get_test_state()
    if state is None:
        return jsonify({"testing": False})

    index = state["question_index"]
    question_count = len(state["questions"])
    return jsonify({
        "testing": True,
        "progress": f"{index + 1}/{question_count}",
        "reply": (
            ("An unfinished assessment was found. We will continue from the current task.\n\n"
             if state.get("ui_language") == "en" else
             "检测到尚未完成的测试，我们从当前题继续。\n\n")
            + question_text(state, index)
        ),
        "question": public_question(state["questions"], index),
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
