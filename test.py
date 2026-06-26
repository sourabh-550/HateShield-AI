# Quick local test — run with: python test_preprocessor.py
import sys
sys.path.append('.')

from src.data.preprocessor import HinglishPreprocessor

p = HinglishPreprocessor()

tests = [
    "Modi ji is best!!!! 😂😂😂",
    "Check this out http://t.co/abc123 @narendramodi #BJP",
    "yaaaar tu bohot zyaadaaa bura hai!!!",
    "I hate you so much!!!! You are worst",
]

for t in tests:
    print(f"IN : {t}")
    print(f"OUT: {p.clean(t)}")
    print()