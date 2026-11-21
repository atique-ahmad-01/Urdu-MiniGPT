"""Audit the raw Urdu dataset: character inventory, suspicious chars, line stats."""
import sys, unicodedata, collections, pathlib
root = pathlib.Path(sys.argv[1])
for name in ["train.txt", "val.txt", "test.txt"]:
    t = (root / name).read_text(encoding="utf-8")
    lines = t.split("\n")
    c = collections.Counter(t)
    cats = collections.Counter(unicodedata.category(ch) for ch in t)
    print(f"== {name}: chars={len(t)} lines={len(lines)} empty={sum(1 for l in lines if not l.strip())} "
          f"maxline={max(map(len,lines))} unique_chars={len(c)}")
    print("  categories:", dict(cats.most_common()))
    groups = collections.Counter()
    odd = collections.Counter()
    for ch, n in c.items():
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F or 0xFB50 <= cp <= 0xFEFF: g = "arabic-script"
        elif ch.isascii() and ch.isalpha(): g = "latin"
        elif ch.isascii() and ch.isdigit(): g = "ascii-digit"
        elif ch.isspace(): g = "space"
        else: g = "other"
        groups[g] += n
        if g == "other" or unicodedata.category(ch) in ("Cf", "Cc", "Mn") or 0xFB50 <= cp <= 0xFEFF:
            odd[ch] += n
    print("  groups:", dict(groups))
    print("  notable chars:", [(f"U+{ord(k):04X}", unicodedata.name(k, "?"), v) for k, v in odd.most_common(25)])
