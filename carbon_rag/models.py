import re
from dataclasses import asdict, dataclass
from datetime import date
from urllib.parse import urlparse

PROVINCES = [
    "北京市",
    "天津市",
    "河北省",
    "山西省",
    "内蒙古自治区",
    "辽宁省",
    "吉林省",
    "黑龙江省",
    "上海市",
    "江苏省",
    "浙江省",
    "安徽省",
    "福建省",
    "江西省",
    "山东省",
    "河南省",
    "湖北省",
    "湖南省",
    "广东省",
    "广西壮族自治区",
    "海南省",
    "重庆市",
    "四川省",
    "贵州省",
    "云南省",
    "西藏自治区",
    "陕西省",
    "甘肃省",
    "青海省",
    "宁夏回族自治区",
    "新疆维吾尔自治区",
    "台湾省",
    "香港特别行政区",
    "澳门特别行政区",
]
DOC_TYPES = ["中国政策", "标准", "国际资料", "研究报告", "企业报告", "译文", "其他参考"]


@dataclass
class Metadata:
    title: str
    issuer: str
    publish_date: str
    source_url: str
    doc_type: str = "中国政策"
    jurisdiction: str = "国家"
    province: str = ""
    verification_status: str = "待核验"
    doc_number: str = ""
    effective_date: str = ""
    validity_note: str = ""
    language: str = "中文"
    is_translation: bool = False
    source_note: str = ""

    def validate(self):
        for key in ("title", "issuer", "publish_date", "source_url", "doc_type", "jurisdiction"):
            if not getattr(self, key).strip():
                raise ValueError(f"缺少必填字段：{key}")
        parsed = urlparse(self.source_url)
        if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username:
            raise ValueError("来源 URL 必须为有效的 HTTP(S) 发布页链接")
        # Some standards and reports publish only a year or month. Keep that
        # precision instead of inventing a day; effective dates stay exact.
        if re.fullmatch(r"\d{4}", self.publish_date):
            pass
        elif re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", self.publish_date):
            pass
        else:
            date.fromisoformat(self.publish_date)
        if self.effective_date:
            date.fromisoformat(self.effective_date)
        if self.doc_type not in DOC_TYPES:
            raise ValueError("未知资料类型")
        if self.jurisdiction not in {"国家", "省级", "国际"}:
            raise ValueError("请选择国家、省级或国际范围")
        if self.jurisdiction == "省级" and self.province not in PROVINCES:
            raise ValueError("省级文件必须填写有效地区")
        if self.jurisdiction != "省级" and self.province:
            raise ValueError("非省级文件的省份应留空")
        if self.doc_type == "中国政策" and (self.is_translation or self.jurisdiction == "国际"):
            raise ValueError("译文、国际资料不能标记为中国政策")
        if self.verification_status not in {"待核验", "已核验"}:
            raise ValueError("核验状态无效")
        return asdict(self)
