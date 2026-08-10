"""日常判断题的语义答案样例库。

样例只用于本地向量相似度比较，不包含任何用户回答。四类标签的含义为：
correct（安全且完整）、partial（方向基本正确但不完整）、incorrect（未回答任务
或无法形成安全行动）以及 dangerous（明确建议危险行动）。
"""

from __future__ import annotations


def _rubric(correct, partial, incorrect, dangerous, *, en_correct, en_partial,
            en_incorrect, en_dangerous):
    return {
        "zh": {
            "correct": tuple(correct),
            "partial": tuple(partial),
            "incorrect": tuple(incorrect),
            "dangerous": tuple(dangerous),
        },
        "en": {
            "correct": tuple(en_correct),
            "partial": tuple(en_partial),
            "incorrect": tuple(en_incorrect),
            "dangerous": tuple(en_dangerous),
        },
    }


SEMANTIC_RUBRICS = {
    "judgment_red_light": _rubric(
        ("我会停下来，等绿灯亮了再过马路", "先站在安全位置等待，确认绿灯后再走", "红灯不能过，要停下等候"),
        ("我会先看看车再决定", "我会小心一点", "先慢下来观察"),
        ("我不知道应该怎么处理", "看情况再说", "我想不起来了"),
        ("没有车就直接走过去", "继续过马路，不用等绿灯", "赶时间就闯过去"),
        en_correct=("I would stop and wait for the green light.", "I would remain somewhere safe until it is green."),
        en_partial=("I would slow down and look around.", "I would be careful."),
        en_incorrect=("I do not know what to do.", "It depends."),
        en_dangerous=("I would keep crossing the road.", "I would go through the red light if no car was coming."),
    ),
    "judgment_smell_gas": _rubric(
        ("我会关闭火源和煤气，开窗通风并找人帮助", "先关掉煤气，远离现场并报警求助", "不能碰电器，我会开窗并请家人处理"),
        ("我会马上找人来看看", "我会先离开厨房", "我会小心检查一下"),
        ("我不知道是不是煤气", "先等等看会不会散掉", "没有想好怎么办"),
        ("我会开灯检查哪里漏气", "点火试试看是不是煤气", "继续做饭，不用理会"),
        en_correct=("I would turn off the gas, open a window and ask for help.", "I would avoid electrical switches and call for help."),
        en_partial=("I would leave the kitchen and ask someone to check.", "I would move away."),
        en_incorrect=("I would wait and see.", "I do not know."),
        en_dangerous=("I would switch on the light to inspect it.", "I would light a flame to check."),
    ),
    "judgment_unknown_medicine": _rubric(
        ("我不会直接吃，会先问医生或家人确认", "先看清药名和医嘱，再联系医生核实", "把药留着，等家属确认后再服用"),
        ("我会先看看药盒说明", "我会把药先放在一边", "我会找个人问一下"),
        ("不知道是什么药", "看起来差不多就行", "我记不清了"),
        ("先吃一颗试试看", "别人给的药可以直接服用", "马上把药吞下去"),
        en_correct=("I would not take it until a doctor or family member confirmed it.", "I would verify the label and prescription first."),
        en_partial=("I would put it aside and check the box.", "I would ask someone."),
        en_incorrect=("I am not sure what it is.", "I cannot remember."),
        en_dangerous=("I would take one pill and see what happens.", "I would swallow it immediately."),
    ),
    "judgment_phone_scam": _rubric(
        ("我不会转账，会先联系家人和银行核实", "先挂断电话，再报警或联系银行确认", "不给对方钱，也不透露密码，找家人商量"),
        ("我会先问问家里人是不是真的", "我先挂电话", "我会查一下对方身份"),
        ("他说得像真的，我不确定", "我不知道该不该相信", "看情况再决定"),
        ("对方催得急就马上转账", "把验证码和银行卡密码告诉他", "按电话要求给陌生账户汇款"),
        en_correct=("I would not pay; I would contact family and the bank to verify it.", "I would hang up and report the call."),
        en_partial=("I would ask my family whether it was real.", "I would check the caller's identity."),
        en_incorrect=("I am not sure whether to believe it.", "It depends on what they say."),
        en_dangerous=("I would transfer the money immediately.", "I would tell the caller my password and verification code."),
    ),
    "judgment_dizzy": _rubric(
        ("我会停下来坐稳，呼叫身边的人帮助", "先在安全地方坐下，再联系家人或医生", "不再独自行走，请别人陪同处理"),
        ("我会先休息一下", "我会扶着东西站稳", "我会告诉旁边的人"),
        ("忍一忍也许就好了", "我不知道为什么头晕", "先看看再说"),
        ("继续一个人走回家", "强撑着快速走过去", "不告诉别人继续活动"),
        en_correct=("I would sit down safely and call someone for help.", "I would stop walking and contact family or a doctor."),
        en_partial=("I would rest for a while.", "I would tell someone nearby."),
        en_incorrect=("I would wait and see.", "I do not know why I feel dizzy."),
        en_dangerous=("I would keep walking alone.", "I would carry on without telling anyone."),
    ),
    "judgment_hot_water": _rubric(
        ("我会先放凉，再小口试温后喝", "等水温合适了再喝", "先检查温度，避免烫伤"),
        ("我会慢慢喝", "我会先等一下", "先用手靠近杯子感受温度"),
        ("不知道水有多热", "和平时一样就行", "看起来能喝"),
        ("刚烧开就大口喝", "不用试温直接喝下去", "趁滚烫马上喝"),
        en_correct=("I would let it cool and test the temperature before drinking.", "I would wait until the water was safe to drink."),
        en_partial=("I would wait for a moment.", "I would drink slowly."),
        en_incorrect=("It looks drinkable.", "I do not know how hot it is."),
        en_dangerous=("I would drink it immediately while boiling hot.", "I would take a large sip without checking."),
    ),
    "judgment_lost": _rubric(
        ("我会留在安全地点，打电话联系家人", "找警察或工作人员求助，不再四处走", "原地等待并告诉家人自己的位置"),
        ("我会找个人问路", "我会拿手机看看地图", "我先停下来想一想"),
        ("随便选个方向走", "不知道这是哪里", "只能碰运气"),
        ("继续一个人到处乱走", "往偏僻的地方找路", "不联系别人一直走下去"),
        en_correct=("I would stay somewhere safe and contact family or the police.", "I would stop wandering and ask staff for help."),
        en_partial=("I would stop and check a map.", "I would ask someone for directions."),
        en_incorrect=("I would choose a direction at random.", "I have no idea where I am."),
        en_dangerous=("I would keep wandering around alone.", "I would walk into a quiet area to find the way."),
    ),
    "judgment_wet_floor": _rubric(
        ("我会放慢脚步，扶稳并绕开湿滑地面", "先停下观察，选择干燥安全的路线", "小心慢走，提醒别人地面湿滑"),
        ("我会慢一点走", "我会看看哪里比较干", "我会扶着旁边"),
        ("应该没什么问题", "和平时一样走", "我没有注意到地面"),
        ("快速跑过去就不会滑倒", "直接冲过湿滑地面", "不用扶也不用减速"),
        en_correct=("I would slow down, hold on and go around the wet area.", "I would choose a dry and safe route."),
        en_partial=("I would walk more slowly.", "I would hold on to something."),
        en_incorrect=("It should be fine.", "I would walk as usual."),
        en_dangerous=("I would run across the wet floor.", "I would rush straight through it."),
    ),
    "judgment_door": _rubric(
        ("我不会直接开门，会先核实身份并联系家人", "隔着门询问来意，确认可靠后再处理", "不让陌生人进屋，必要时报警"),
        ("我会先问他是谁", "我会从猫眼看看", "我会打电话问家人"),
        ("他说认识我，应该可以吧", "我不知道来的人是谁", "看起来不像坏人"),
        ("马上打开门让陌生人进来", "不用核实身份就开门", "把门打开再问他是谁"),
        en_correct=("I would keep the door closed, verify the person and contact family.", "I would not let a stranger in and would call for help if needed."),
        en_partial=("I would ask who they were through the door.", "I would look through the peephole."),
        en_incorrect=("They probably look trustworthy.", "I do not know who they are."),
        en_dangerous=("I would open the door and let the stranger in.", "I would open it before checking their identity."),
    ),
    "judgment_power": _rubric(
        ("我会远离电线，断电后请专业人员处理", "不直接触碰，提醒别人避开并报警求助", "保持安全距离，找物业或维修人员"),
        ("我会去找人帮忙", "我会提醒别人小心", "我会站远一点"),
        ("不知道电线有没有电", "看起来应该没事", "先放在那里不管"),
        ("用手把电线捡起来", "直接摸一下看看有没有电", "自己拿金属工具去处理"),
        en_correct=("I would stay away, turn off the power if safe, and call a professional.", "I would not touch the wire and would warn other people."),
        en_partial=("I would move away and ask someone for help.", "I would warn others."),
        en_incorrect=("It probably is not live.", "I would leave it there without warning anyone."),
        en_dangerous=("I would pick up the wire with my hand.", "I would touch it to see whether it was live."),
    ),
}


def examples_for(question_id: str, language: str = "zh") -> dict[str, tuple[str, ...]]:
    """返回指定题目的静态样例；不回传或保存用户原话。"""
    rubric = SEMANTIC_RUBRICS.get(question_id)
    if rubric is None:
        raise KeyError(f"缺少语义样例库: {question_id}")
    language = "en" if str(language).lower() in {"en", "english"} else "zh"
    return rubric[language]
