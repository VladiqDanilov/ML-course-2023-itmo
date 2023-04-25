from math import log, exp

# Данные:
spam = 14
not_spam = 12
spam_words = 70
not_spam_words = 46
words = {
    'Bill': (1, 1),
    'Million': (2, 1),
    'Gift': (1, 4),
    'Money': (0, 1),
    'Access': (5, 4),
    'Free': (8, 12),
    'Purchase': (14, 22),
    'Cash': (31, 1),
    'Coupon': (7, 0),
    'Online': (1, 0)
}
text = 'Gift Purchase Offer Access Money Million Investment'.split(' ')

print('P("спам") =', spam / (spam + not_spam))

f_spam = log(spam / (spam + not_spam))
f_not_spam = log(not_spam / (spam + not_spam))

r = len(list(filter(lambda x: x not in words, text)))

for word in text:
    a, b = words.get(word, (0, 0))
    
    f_spam += log((1 + a) / (spam_words + len(words) + r))
    f_not_spam += log((1 + b) / (not_spam_words + len(words) + r))

print('F("спам") =', f_spam)
print('F("не спам") =', f_not_spam)
print('P("спам"|Письмо) =', exp(f_spam) / (exp(f_spam) + exp(f_not_spam)))