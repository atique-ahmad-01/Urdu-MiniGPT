"""Check Urdu-specific issues: آ presence, Arabic vs Urdu letter variants, digits, punctuation."""
import sys, pathlib, re, collections
root = pathlib.Path(sys.argv[1])
checks = {
 "U+0622 آ ALEF MADDA": "آ", "U+0627+U+0653 ا+ٓ (decomposed آ)": "آ",
 "U+064A ي Arabic YEH": "ي", "U+06CC ی Farsi YEH (Urdu)": "ی", "U+0649 ى Alef maksura": "ى",
 "U+0643 ك Arabic KAF": "ك", "U+06A9 ک Keheh (Urdu)": "ک",
 "U+0647 ه Arabic HEH": "ه", "U+06C1 ہ Heh goal (Urdu)": "ہ", "U+06BE ھ Do-chashmi": "ھ",
 "U+06D2 ے Yeh barree": "ے", "U+0624 ؤ": "ؤ", "U+0626 ئ": "ئ",
 "U+06D4 ۔ Urdu full stop": "۔", "U+060C ، comma": "،", "U+061F ؟ question": "؟",
 "U+066B/066C arabic sep": "٫", "U+200C ZWNJ": "‌", "U+200D ZWJ": "‍", "U+00A0 NBSP": " ",
 "U+FEFF BOM": "﻿", "TAB": "\t", "CR": "\r",
}
for name in ["train.txt", "val.txt", "test.txt"]:
    t = (root / name).read_text(encoding="utf-8")
    print(f"== {name}")
    for k, s in checks.items():
        n = t.count(s)
        if n: print(f"  {k}: {n}")
    for label, rng in [("Arabic-Indic digits ٠-٩", "٠-٩"), ("Urdu digits ۰-۹", "۰-۹"), ("ASCII digits", "0-9")]:
        print(f"  {label}: {len(re.findall(f'[{rng}]', t))}")
    print("  double spaces:", t.count("  "), "| leading/trailing-space lines:", sum(1 for l in t.split('\n') if l != l.strip()))
    # words starting with ئ that likely lost آ, e.g. ' ئے'
    print("  ' ئے ' occurrences (likely lost آ):", t.count(" ئے "), "| ' غاز' :", t.count(" غاز"), "| ' پ ' (آپ->پ?):", t.count(" پ "))
    print("  sentences ending '۔' per line avg:", round(t.count('۔')/max(1,t.count('\n')),2))
t = (root/"train.txt").read_text(encoding="utf-8")
w = collections.Counter(t.split())
print("\nTop 40 train words:", [x for x,_ in w.most_common(40)])
print("Unique words:", len(w), " total words:", sum(w.values()))
print("Latin snippets:", re.findall(r"[A-Za-z][A-Za-z ]{3,30}", t)[:15])
