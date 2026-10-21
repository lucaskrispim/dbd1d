# SPDX-License-Identifier: GPL-3.0-or-later
"""Ar seco (N2 78%, O2 21%, Ar 1%) montado do mecanismo N2-O2 do ZDPlasKin
(Pancheshnyi, versão 1.03) na adaptação do usuário para as simulações air2D
(original/, md5 ef36e221 e 08eb361e: processos sem seção de choque comentados;
53 espécies, íons negativos, NxOy, blocos @B/@R e DENSITY no BOLSIG+) e do
mecanismo do argônio da raiz, mais seis
reações cruzadas com coeficientes típicos da literatura (aqui servem ao teste
da ferramenta numa mistura nova, não a um estudo do ar):
  Ar^+ + N2 => N2^+ + Ar          1,0e-11 cm3/s
  Ar^+ + O2 => O2^+ + Ar          4,9e-11 cm3/s
  Ar2^+ + N2 => N2^+ + Ar + Ar    1,0e-11 cm3/s
  Ar2^+ + O2 => O2^+ + Ar + Ar    1,0e-10 cm3/s
  Ar* + N2 => Ar + N2(C3)         3,0e-11 cm3/s
  Ar* + O2 => Ar + O + O          2,1e-10 cm3/s
Uso: python3 casos/ar_n2_o2/monta.py   (grava kinet.inp, bolsigdb.dat e composicao.txt aqui)
"""
import pathlib
import re

AQUI = pathlib.Path(__file__).resolve().parent
RAIZ = AQUI.parent.parent

ar_seco = (AQUI / "original" / "kinet_ar_air2D.inp").read_text()
argonio = (RAIZ / "kinet.inp").read_text()


def troca_uma(texto, padrao, novo):
    texto, n = re.subn(padrao, novo, texto, count=1, flags=re.M | re.S)
    assert n == 1, padrao
    return texto


reacoes_ar = re.search(r"^REACTIONS\s*\n(.*)^END", argonio, re.S | re.M).group(1).rstrip()
kinet = troca_uma(ar_seco, r"^ELEMENTS\s*\ne N O\s*\nEND", "ELEMENTS\ne N O Ar\nEND")
kinet = troca_uma(kinet, r"^(BOLSIG\s*\n.*?)^END", r"\1Ar Ar*\nEND")
kinet = troca_uma(kinet, r"^(SPECIES\s*\n.*?)^END", r"\1Ar Ar* Ar^+ Ar2^+\nEND")
cruzadas = """
#==================================================================================================================================
# argônio (kinet.inp da raiz) e reações cruzadas (casos/ar_n2_o2/monta.py)
#==================================================================================================================================
""" + reacoes_ar + """

Ar^+ + N2 => N2^+ + Ar             !   1.0d-11
Ar^+ + O2 => O2^+ + Ar             !   4.9d-11
Ar2^+ + N2 => N2^+ + Ar + Ar       !   1.0d-11
Ar2^+ + O2 => O2^+ + Ar + Ar       !   1.0d-10
Ar* + N2 => Ar + N2(C3)            !   3.0d-11
Ar* + O2 => Ar + O + O             !   2.1d-10
"""
# antes do END do bloco REACTIONS (o arquivo tem outro END num comentário final)
m = re.search(r"^REACTIONS\s*\n.*?^END", kinet, re.S | re.M)
kinet = kinet[:m.end() - 3] + cruzadas + "\n" + kinet[m.end() - 3:]
(AQUI / "kinet.inp").write_text(kinet)
(AQUI / "bolsigdb.dat").write_text((AQUI / "original" / "bolsigdb_ar_air2D.dat").read_text().rstrip() + "\n\n"
                                   + (RAIZ / "bolsigdb.dat").read_text())
(AQUI / "composicao.txt").write_text("N2 0.78\nO2 0.21\nAr 0.01\n")
print("kinet.inp, bolsigdb.dat e composicao.txt gravados")
