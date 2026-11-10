# =============================================================================
#  teshis_gradyan.py — int8 tıkanma teşhisi, Adım 2: int8 gradyanı gerçek (float) gradyanla ne kadar aynı yönde?
# -----------------------------------------------------------------------------
#  Aynı ağırlıklar ve aynı batch'le:
#    int  : hizli_model.egitim_adimi (golden = RTL ile bit bit) -> E[l] (int8 hata), Gw (int32 WG), g8 (m_u bitlik güncelleme)
#    float: aynı ağırlıkların float karşılığı (W * 2^sw), autograd -> dL/dz (her conv çıkışı, ReLU öncesi) ve dL/dW
#  Katman başına kosinüs benzerliği:
#    cos_E  = cos(E_int[l], dL/dz_float[l])      hata sinyalinin yönü (BP zincirinde birikmiş kayıp)
#    cos_Gw = cos(Gw_int, dL/dW_float)           m_u yuvarlamasından ÖNCE ağırlık gradyanı
#    cos_g8 = cos(g8, dL/dW_float)               uygulanan güncellemenin yönü
#  Ayrıca: E'nin sıfır olmayan oranı (ReLU'su açık konumlarda), g8'in sıfır olmayan oranı.
#  --adim N: N gerçek eğitim adımı boyunca katman başına net hareket oranı |sum dW| / sum |dW|
#  (1'e yakın: güncellemeler aynı yönde birikiyor; 0'a yakın: gidip geliyor).
#  Kullanım:  python teshis_gradyan.py [--durum sonuc/vgg16like_int/durum.pt] [--mu 2] [--batch_sayisi 4] [--adim 100]
#  Yazan: Claude, 10.11.2026
# =============================================================================
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import egit as E
import hizli_model as H


def cos(a, b):
    a, b = a.double().flatten(), b.double().flatten()
    n = a.norm() * b.norm()
    return float("nan") if n == 0 else (a @ b / n).item()


def float_ag(K, ad):
    """K'nin ağırlıklarını (W * 2^sw) taşıyan FloatAg."""
    m = E.FloatAg(ad).to(H.DEV).double()
    convlar = [x for x in m.ag if isinstance(x, (nn.Conv2d, nn.Linear))]
    agirlikli = [k for k in K if k.tur in ("conv", "fc")]
    assert len(convlar) == len(agirlikli)
    with torch.no_grad():
        for c, k in zip(convlar, agirlikli):
            c.weight.copy_(k.W.double().view_as(c.weight) * 2.0 ** k.sw)
    return m, convlar


def float_gradyan(m, convlar, x, y):
    """dL/dz her conv çıkışı için (ReLU öncesi) ve dL/dW."""
    zler = []
    def kanca(mod, giris, cikis):
        cikis.retain_grad(); zler.append(cikis)
    h = [c.register_forward_hook(kanca) for c in convlar]
    m.zero_grad()
    kayip = F.cross_entropy(m(x.double()), y)
    kayip.backward()
    for hh in h:
        hh.remove()
    return [z.grad for z in zler], [c.weight.grad for c in convlar], kayip.item()


def olc(K, ad, x, y, m_u):
    K_yedek = [None if k.W is None else k.W.clone() for k in K]
    A0, s0 = E.float_to_int8(x)
    r = H.egitim_adimi(K, A0, s0, y, m_u)
    for k, W in zip(K, K_yedek):                        # ölçüm: ağırlıkları geri al
        k.W = W
    m, convlar = float_ag(K, ad)
    dz, dW, kayip = float_gradyan(m, convlar, A0.double() * 2.0 ** s0, y)
    sat, j = [], -1
    for li, k in enumerate(K):
        if k.tur not in ("conv", "fc"):
            continue
        j += 1
        El = r["E"][li + 1]                              # katman li'nin çıkışındaki hata (maskeli)
        X = r["A"][li]
        if k.tur == "fc":
            N = X.shape[0]
            Gw = H.int_mm(El.reshape(N, -1).t().contiguous(), X.reshape(N, -1))
            g8, _ = H.sru_katman(Gw.t().contiguous(), m_u)
            g8 = g8.t()
        else:
            Gw = H.conv_wg(X, El, k.s, k.p)
            g8, _ = H.sru_wg(Gw, m_u)
        acik = (r["A"][li + 1] > 0) if k.relu else torch.ones_like(El, dtype=torch.bool)
        dWj = dW[j].view_as(Gw)
        sat.append(dict(cos_E=cos(El, dz[j].view_as(El)), cos_Gw=cos(Gw, dWj), cos_g8=cos(g8, dWj),
                        E_nz=((El != 0) & acik).double().sum().item() / max(1, acik.sum().item()),
                        g8_nz=(g8 != 0).double().mean().item()))
    return sat, kayip


def net_hareket(K, yuk, m_u, n):
    A = [k for k in K if k.W is not None]
    W0 = [k.W.clone() for k in A]
    toplam = [torch.zeros_like(w, dtype=torch.float64) for w in W0]
    mutlak = [torch.zeros_like(w, dtype=torch.float64) for w in W0]
    i = 0
    while i < n:
        for x, y in yuk:
            x, y = x.to(H.DEV), y.to(H.DEV)
            once = [k.W.clone() for k in A]
            A0, s0 = E.float_to_int8(x)
            H.egitim_adimi(K, A0, s0, y, m_u)
            for j, k in enumerate(A):
                d = (k.W - once[j]).double()
                toplam[j] += d; mutlak[j] += d.abs()
            i += 1
            if i >= n:
                break
    for k, w in zip(A, W0):
        k.W = w
    return [(t.abs().sum() / max(1e-9, a.sum().item())).item() for t, a in zip(toplam, mutlak)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ag", default="vgg16like")
    ap.add_argument("--durum", default="", help="boşsa tohumla başlangıç ağırlıkları")
    ap.add_argument("--tohum", type=int, default=1)
    ap.add_argument("--mu", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--batch_sayisi", type=int, default=4)
    ap.add_argument("--adim", type=int, default=0)
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    K = E.int_ag_kur(a.ag, a.tohum)
    etiket = f"başlangıç (tohum {a.tohum})"
    if a.durum:
        d = torch.load(a.durum, map_location="cpu", weights_only=False)
        for k, (W, sw) in zip(K, d["agirlik"]):
            if W is not None:
                k.W, k.sw = W.to(H.DEV), sw
        etiket = f"{a.durum} (epoch {d['epoch']})"
    for k in K:
        if k.W is not None:
            k.W = k.W.to(H.DEV)
    torch.manual_seed(123)
    yuk, _ = E.veri(a.batch, 0)
    birik, kayiplar = None, []
    for bi, (x, y) in enumerate(yuk):
        if bi >= a.batch_sayisi:
            break
        sat, kayip = olc(K, a.ag, x.to(H.DEV), y.to(H.DEV), a.mu)
        kayiplar.append(kayip)
        birik = sat if birik is None else [{key: b[key] + s[key] for key in b} for b, s in zip(birik, sat)]
    n = len(kayiplar)
    print(f"\n== {etiket}, m_u {a.mu}, {n} batch × {a.batch}; float kaybı ort {sum(kayiplar) / n:.3f} ==")
    print(f"{'kat':>3} {'cos_E':>7} {'cos_Gw':>7} {'cos_g8':>7} {'E≠0':>7} {'g8≠0':>7}")
    for li, s in enumerate(birik):
        print(f"{li + 1:>3} {s['cos_E'] / n:>7.3f} {s['cos_Gw'] / n:>7.3f} {s['cos_g8'] / n:>7.3f} "
              f"{s['E_nz'] / n:>7.3f} {s['g8_nz'] / n:>7.3f}")
    if a.adim:
        r = net_hareket(K, yuk, a.mu, a.adim)
        print(f"\nnet hareket oranı |ΣΔW|/Σ|ΔW|, {a.adim} adım:")
        print("  " + "  ".join(f"{i + 1}:{v:.3f}" for i, v in enumerate(r)))


if __name__ == "__main__":
    main()
