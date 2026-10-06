"""Mapping MCC -> kategori budget (bahasa nasabah).

Mapping ini disusun dan direview manual (109 MCC di dataset). Untuk MCC baru yang
belum ada di tabel, `llm_advisor.classify_new_mcc()` dapat dipakai untuk usulan kategori
(LLM gpt-6-luna) yang kemudian direview analis sebelum masuk tabel ini.
"""

CATEGORY_META = {
    # kode: (label Indonesia, tipe kebutuhan)
    "groceries":      ("Belanja Kebutuhan Pokok", "essential"),
    "transport":      ("Transportasi & BBM", "essential"),
    "bills":          ("Tagihan, Utilitas & Asuransi", "essential"),
    "health":         ("Kesehatan & Apotek", "essential"),
    "dining":         ("Makan & Minum di Luar", "discretionary"),
    "shopping":       ("Belanja Ritel & Rumah Tangga", "discretionary"),
    "entertainment":  ("Hiburan & Gaya Hidup", "discretionary"),
    "travel":         ("Travel & Akomodasi", "discretionary"),
    "transfer":       ("Transfer Uang", "transfer"),
    "services":       ("Jasa & Lainnya", "other"),
}

_MAP = {
    "groceries": [5411, 5499, 5300],
    "transport": [5541, 4784, 4121, 4111, 3722, 3771, 4112, 4131, 7538, 7542, 5533, 7531, 7549, 4511],
    "bills": [4900, 4814, 4899, 6300],
    "health": [5912, 8021, 8011, 8062, 8043, 8099, 8041, 8049],
    "dining": [5812, 5814, 5813, 5921],
    "shopping": [5311, 5310, 5651, 5719, 5211, 5251, 5712, 5732, 5661, 5655, 5621, 5045, 5722,
                 5733, 5977, 5941, 5947, 5094, 5932, 3144, 3174, 5261, 5193, 5970, 5942, 5192,
                 3132, 3260, 3256, 3640, 3504],
    "entertainment": [7832, 7922, 7801, 7802, 7996, 5815, 5816, 7230, 7995],
    "travel": [4722, 7011, 4411],
    "transfer": [4829],
    "services": [8111, 7276, 8931, 9402, 4214, 7349, 7210, 7393, 1711, 3775, 3730, 3509, 3684,
                 3780, 3596, 3001, 3058, 3000, 3066, 3359, 3389, 3387, 3390, 3395, 3405, 3393,
                 3009, 3008, 3007, 3005, 3006, 3075],
}

MCC_TO_CATEGORY = {mcc: cat for cat, mccs in _MAP.items() for mcc in mccs}
CATEGORIES = list(CATEGORY_META.keys())
GAMBLING_MCC = [7995]


def category_of(mcc: int) -> str:
    return MCC_TO_CATEGORY.get(int(mcc), "services")
