# =============================================================================
#  teshis_agirlik.py — int8 tıkanma teşhisi, Adım 1 (eğitimsiz): katman katman ağırlık değişimi
# -----------------------------------------------------------------------------
#  Başlangıç ağırlıkları (aynı tohumla yeniden üretilir) ile bir koşunun durum.pt'sindeki ağırlıkları karşılaştırır.
#  Katman başına: göreli değişim ||W1-W0||/||W0||, değişen eleman oranı, W0-W1 kosinüsü, ±127'deki eleman oranı.
#  Kosinüs ~1: katman başlangıçtan pek ayrılmamış; ~0: başlangıçla ilişkisi kalmamış.
#  Kullanım:  python teshis_agirlik.py sonuc/vgg16like_int [sonuc/vgg16like_float30]
#  Yazan: Claude, 10.11.2026
# =============================================================================
import sys
import torch
import egit as E


def satirlar(W0, W1):
    W0, W1 = W0.double().flatten(), W1.double().flatten()
    d = W1 - W0
    return (d.norm() / W0.norm()).item(), (d != 0).double().mean().item(), \
        torch.nn.functional.cosine_similarity(W0, W1, 0).item()


def int_rapor(klasor, tohum=1):
    d = torch.load(f"{klasor}/durum.pt", map_location="cpu", weights_only=False)
    K0 = E.int_ag_kur(d["ag"], tohum)
    print(f"\n== int: {klasor} (epoch {d['epoch']}) ==")
    print(f"{'kat':>3} {'boyut':>16} {'sw0':>4} {'sw1':>4} {'göreli':>7} {'değişen':>8} {'cos':>6} {'±127 ilk':>9} {'±127 son':>9} {'|W| ort ilk/son':>16}")
    for i, (k, (W1, sw1)) in enumerate(zip(K0, d["agirlik"])):
        if W1 is None:
            continue
        W0 = k.W.cpu()
        W1 = W1.cpu()
        g, deg, cos = satirlar(W0, W1)
        s0 = (W0.abs() >= 127).double().mean().item()
        s1 = (W1.abs() >= 127).double().mean().item()
        print(f"{i + 1:>3} {str(tuple(W0.shape)):>16} {k.sw:>4} {sw1:>4} {g:>7.3f} {deg:>8.3f} {cos:>6.3f} "
              f"{s0:>9.4f} {s1:>9.4f} {W0.abs().double().mean().item():>7.2f}/{W1.abs().double().mean().item():<7.2f}")


def float_rapor(klasor, tohum=1):
    d = torch.load(f"{klasor}/durum.pt", map_location="cpu", weights_only=False)
    torch.manual_seed(tohum)
    m0 = E.FloatAg(d["ag"])
    s1 = d["model"]
    print(f"\n== float: {klasor} (epoch {d['epoch']}) ==")
    print(f"{'kat':>3} {'boyut':>16} {'göreli':>7} {'cos':>6}")
    i = 0
    for ad, W0 in m0.state_dict().items():
        if not ad.endswith("weight"):
            continue
        i += 1
        g, _, cos = satirlar(W0, s1[ad].cpu())
        print(f"{i:>3} {str(tuple(W0.shape)):>16} {g:>7.3f} {cos:>6.3f}")


if __name__ == "__main__":
    int_rapor(sys.argv[1])
    if len(sys.argv) > 2:
        float_rapor(sys.argv[2])
