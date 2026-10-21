from scripts.export_catalog import final_records
from scripts.import_catalog import classify, metadata_for, resolve_file


def record(row, region="全国", date="2024-01-01"):
    return {
        "sheet_row": row,
        "region": region,
        "title": "测试资料",
        "issuer": "测试机构",
        "publish_date": date,
        "source_url": "https://example.com/source",
        "note": "",
    }


def test_filename_whitespace_match_and_conflicting_duplicates(tmp_path):
    a = tmp_path / "指南水泥行业.pdf"
    a.write_bytes(b"same")
    selected, match = resolve_file("指南 水泥行业.pdf", [a])
    assert selected == a
    assert match == "忽略空格后匹配"
    duplicate = tmp_path / "copy" / a.name
    duplicate.parent.mkdir()
    duplicate.write_bytes(b"different")
    assert resolve_file(a.name, [a, duplicate]) == (None, "同名文件内容不一致，需选择版本")


def test_catalog_classification_and_city_scope():
    assert classify(record(13)) == ("国家", "", "研究报告")
    assert classify(record(27)) == ("国家", "", "其他参考")
    assert classify(record(50, "国际")) == ("国际", "", "标准")
    meta = metadata_for(record(43, "深圳市"), [3])
    assert meta.jurisdiction == "省级"
    assert meta.province == "广东省"
    assert "适用城市：深圳市" in meta.source_note
    assert "1 页无法提取正文" in meta.source_note


def test_final_catalog_excludes_only_withdrawn_pending_rows():
    rows = [
        {"sheet_row": 2, "check_result": "发布页与 PDF 一致"},
        {"sheet_row": 41, "check_result": "待核对"},
        {"sheet_row": 43, "check_result": "格式待转换"},
    ]
    assert [item["sheet_row"] for item in final_records(rows)] == [2, 43]
