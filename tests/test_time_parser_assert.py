# -*- coding: utf-8 -*-
"""时间解析器 - 关键断言测试"""
import sys
import os
import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from memory.storage import parse_time_expression

NOW = datetime.datetime(2026, 2, 11, 14, 30, 0)  # 星期三
TODAY = NOW.date()
passed = 0
failed = 0

def assert_test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}  {detail}")

# 1. 上周 → 2026-02-02(周一) ~ 2026-02-08(周日)
r = parse_time_expression("上周我家的小猫吃了两根火腿", now=NOW)
assert_test("上周-时间范围起", 
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2026-02-02",
    f'got {datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d")}')
assert_test("上周-时间范围止", 
    datetime.datetime.fromtimestamp(r["time_range"][1]).strftime("%Y-%m-%d") == "2026-02-08")
assert_test("上周-时间表达式", r["time_expr"] == "上周")
assert_test("上周-关键词含小猫", "小猫" in r["keywords"])
assert_test("上周-关键词含火腿", "火腿" in r["keywords"])

# 2. 去年 → 2025全年
r = parse_time_expression("去年汽车维修的保养费是300美元", now=NOW)
assert_test("去年-时间表达式", r["time_expr"] == "去年")
assert_test("去年-关键词含维修", "维修" in r["keywords"] or "汽车维修" in r["keywords"])
assert_test("去年-300不是时间", "300" not in (r["time_expr"] or ""))

# 3. 前两个月 → 2025-12-11 ~ 2026-02-11
r = parse_time_expression("前两个月你才刚作完的手术", now=NOW)
assert_test("前两个月-时间表达式", r["time_expr"] == "前两个月")
assert_test("前两个月-起始日期", 
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2025-12-11")
assert_test("前两个月-关键词含手术", "手术" in r["keywords"])

# 4. 今年5月份到8月份 → 2026-05-01 ~ 2026-08-31
r = parse_time_expression("今年5月份到8月份的山姆圣诞赠礼活动", now=NOW)
assert_test("月范围-时间表达式", r["time_expr"] == "今年5月份到8月份")
assert_test("月范围-起始5月", 
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m") == "2026-05")
assert_test("月范围-结束8月",
    datetime.datetime.fromtimestamp(r["time_range"][1]).strftime("%Y-%m") == "2026-08")
assert_test("月范围-关键词含山姆", "山姆" in r["keywords"])

# 5. 今天 → 2026-02-11
r = parse_time_expression("今天发生了什么事", now=NOW)
assert_test("今天-时间表达式", r["time_expr"] == "今天")
assert_test("今天-日期正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2026-02-11")

# 6. 前天 → 2026-02-09（不被 before_n 误匹配）
r = parse_time_expression("前天的会议记录", now=NOW)
assert_test("前天-时间表达式", r["time_expr"] == "前天")
assert_test("前天-日期正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2026-02-09")

# 7. 前年 → 2024年（不被 before_n 误匹配）
r = parse_time_expression("前年的事情", now=NOW)
assert_test("前年-时间表达式", r["time_expr"] == "前年")
assert_test("前年-年份正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y") == "2024")

# 8. 300美元 → 无时间匹配
r = parse_time_expression("300美元的汽车保养", now=NOW)
assert_test("300美元-无时间匹配", r["time_range"] is None)
assert_test("300美元-无时间表达式", r["time_expr"] is None)

# 9. 最近三天 → 2026-02-08 ~ 2026-02-11
r = parse_time_expression("最近三天的新闻", now=NOW)
assert_test("最近三天-时间表达式", r["time_expr"] == "最近三天")
assert_test("最近三天-起始日期",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2026-02-08")

# 10. 两个月前 → 同"前两个月"
r = parse_time_expression("两个月前买的车", now=NOW)
assert_test("N月前-时间表达式", r["time_expr"] == "两个月前")
assert_test("N月前-起始12月",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2025-12-11")

# 11. 2024年 → 整年
r = parse_time_expression("2024年的旅行计划", now=NOW)
assert_test("2024年-时间表达式", r["time_expr"] == "2024年")
assert_test("2024年-年份正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y") == "2024")

# 12. 空输入
r = parse_time_expression("", now=NOW)
assert_test("空输入-无时间", r["time_range"] is None)

r = parse_time_expression(None, now=NOW)
assert_test("None输入-无时间", r["time_range"] is None)

# 13. 今年5月3日 → 特定日期
r = parse_time_expression("今年5月3日的约会", now=NOW)
assert_test("年月日-时间表达式", r["time_expr"] == "今年5月3日")
assert_test("年月日-日期正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2026-05-03")
assert_test("年月日-关键词含约会", "约会" in r["keywords"])

# 14. 上个月 → 2026年1月
r = parse_time_expression("上个月的账单", now=NOW)
assert_test("上个月-时间表达式", r["time_expr"] == "上个月")
assert_test("上个月-月份正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m") == "2026-01")

# 15. 上上周
r = parse_time_expression("上上周的会议", now=NOW)
assert_test("上上周-时间表达式", r["time_expr"] == "上上周")
assert_test("上上周-起始日期",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2026-01-26")

# 16. 本周 → 2026-02-09(周一) ~ 2026-02-15(周日)
r = parse_time_expression("本周的工作安排", now=NOW)
assert_test("本周-时间表达式", r["time_expr"] == "本周")
assert_test("本周-起始周一",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m-%d") == "2026-02-09")

# 17. 本月 → 2026年2月
r = parse_time_expression("本月的开支明细", now=NOW)
assert_test("本月-时间表达式", r["time_expr"] == "本月")
assert_test("本月-月份正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m") == "2026-02")

# 18. 今年5月 → 单月
r = parse_time_expression("今年5月的考试", now=NOW)
assert_test("今年5月-时间表达式", r["time_expr"] == "今年5月")
assert_test("今年5月-月份正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m") == "2026-05")

# 19. 去年12月份
r = parse_time_expression("去年12月份发生的事", now=NOW)
assert_test("去年12月-时间表达式", r["time_expr"] == "去年12月份")
assert_test("去年12月-月份正确",
    datetime.datetime.fromtimestamp(r["time_range"][0]).strftime("%Y-%m") == "2025-12")

# ============================================================
# 置信度系统测试 — 误触防护
# ============================================================

print(f"\n{'='*60}")
print("  Confidence System Tests — Anti-False-Positive")
print(f"{'='*60}")

# -- 负向场景: 时间词应被识别但 confidence 太低，time_range=None --

# C1. 描述动词 "提到" → 不应解析为约束
r = parse_time_expression("那篇文章里提到上周的市场波动", now=NOW)
assert_test("C1-描述动词-time_expr识别", r["time_expr"] == "上周")
assert_test("C1-描述动词-time_range为None", r["time_range"] is None)
assert_test("C1-描述动词-confidence<0.45", r["confidence"] is not None and r["confidence"] < 0.45)
assert_test("C1-描述动词-原文保留", r["cleaned_text"] == "那篇文章里提到上周的市场波动")

# C2. 描述动词 "分析了" + "中" → 不应解析
r = parse_time_expression("报告中分析了2024年的GDP数据", now=NOW)
assert_test("C2-分析报告-time_expr识别", r["time_expr"] == "2024年")
assert_test("C2-分析报告-time_range为None", r["time_range"] is None)
assert_test("C2-分析报告-confidence<0.45", r["confidence"] is not None and r["confidence"] < 0.45)

# C3. 书名号内 → 不应解析
r = parse_time_expression("《2025年经济展望》这本书很不错", now=NOW)
assert_test("C3-书名号-time_expr识别", r["time_expr"] == "2025年")
assert_test("C3-书名号-time_range为None", r["time_range"] is None)
assert_test("C3-书名号-有引号内信号", "-引号内" in r["confidence_signals"])

# C4. 描述动词 "说了" → 不应解析
r = parse_time_expression("他说了去年发生的一件事情", now=NOW)
assert_test("C4-说了去年-time_expr识别", r["time_expr"] == "去年")
assert_test("C4-说了去年-time_range为None", r["time_range"] is None)

# C5. "文章里" + "讨论" → 不应解析
r = parse_time_expression("文章里讨论今年5月的数据变化趋势", now=NOW)
assert_test("C5-文章讨论-time_expr识别", r["time_expr"] == "今年5月")
assert_test("C5-文章讨论-time_range为None", r["time_range"] is None)

# -- 正向场景: 查询动词或句首 → 应正常解析 --

# C6. 查询动词 "帮我找" → 应解析
r = parse_time_expression("帮我找上周的会议记录", now=NOW)
assert_test("C6-帮我找-time_range有值", r["time_range"] is not None)
assert_test("C6-帮我找-confidence>=0.45", r["confidence"] >= 0.45)
assert_test("C6-帮我找-有查询动词信号", "+查询动词" in r["confidence_signals"])

# C7. 查询动词 "搜索" → 应解析
r = parse_time_expression("搜索2024年的报告", now=NOW)
assert_test("C7-搜索-time_range有值", r["time_range"] is not None)
assert_test("C7-搜索-confidence>=0.45", r["confidence"] >= 0.45)

# C8. 查询动词 "查看" → 应解析
r = parse_time_expression("查看最近三天的日志", now=NOW)
assert_test("C8-查看-time_range有值", r["time_range"] is not None)
assert_test("C8-查看-confidence>=0.45", r["confidence"] >= 0.45)

# C9. 句首时间词（典型查询） → 应解析
r = parse_time_expression("昨天下午的进展如何", now=NOW)
assert_test("C9-句首昨天-time_range有值", r["time_range"] is not None)
assert_test("C9-句首昨天-confidence>=0.45", r["confidence"] >= 0.45)
assert_test("C9-句首昨天-有句首信号", "+句首" in r["confidence_signals"])

# C10. confidence_threshold=0.0 兼容旧行为（禁用过滤）
r = parse_time_expression("那篇文章里提到上周的市场波动", now=NOW,
                          confidence_threshold=0.0)
assert_test("C10-阈值0-time_range有值", r["time_range"] is not None,
            "confidence_threshold=0.0 应禁用过滤")

# C11. 中文双引号内 → 不应解析
r = parse_time_expression('他引用了\u201c去年的数据\u201d来说明问题', now=NOW)
assert_test("C11-中文引号-time_expr识别", r["time_expr"] == "去年")
assert_test("C11-中文引号-time_range为None", r["time_range"] is None)

# C12. 无匹配时 confidence 应为 None
r = parse_time_expression("这是一段没有时间词的文本", now=NOW)
assert_test("C12-无匹配-confidence为None", r["confidence"] is None)
assert_test("C12-无匹配-signals为None", r["confidence_signals"] is None)

print(f"\n{'='*60}")
print(f"  结果: {passed} passed, {failed} failed, {passed+failed} total")
print(f"{'='*60}")

if failed > 0:
    exit(1)
else:
    print("  ALL TESTS PASSED!")
