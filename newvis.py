matches = {}
maxValue = 0
signStr = ''
for sign in potentialSigns:
    id = str(uuid.uuid4())[:4]
    matches[id] = {}
    for name in templates.keys():
        temp = cv2.resize(templates[name], (sign.shape[1], sign.shape[0]))

        res = cv2.matchTemplate(sign, templates[name], cv2.TM_CCOEFF_NORMED)
        matches[id][name] = np.max(res)
        if np.max(res) > maxValue:
            maxValue = np.max(res)
            signStr = name

print(str(matches))
# cv2.imshow('blur', blur)
cv2.putText(frame, signStr, (100, 100), cv2.QT_FONT_NORMAL, 2, (0, 0, 0), 2, cv2.LINE_8)