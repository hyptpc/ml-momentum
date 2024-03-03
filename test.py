import uproot

file = uproot.open("../root/gen7208.root")
tout = file["g4hyptpc"] #TName
tout.show()
print(tout["HTOF"].array())