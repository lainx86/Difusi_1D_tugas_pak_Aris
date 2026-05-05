# Tugas Pemodelan Oseanografi
# Dosen Pengampu: Dr. Aris Ismanto, S.Si., M.Si.
# Persmanaan Difusi 1-D
# Feby Syarief Al A`raaf` | 26050124130087 | Oseanografi C

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


NAMA  = "Feby Syarief A"
NIM   = "26050124130087"
KELAS = "Oseanografi C"

DX    = 0.1
AD    = 1.0
T_END = 0.1

CASES = {
    "Kasus_I":   0.001,
    "Kasus_II":  0.005,
    "Kasus_III": 0.01,
}

CHOSEN_TIMES = [0.0, 0.01, 0.02, 0.05, 0.1]
X_COMPARE    = 0.3


def get_identitas_judul(nama: str, nim: str, kelas: str) -> str:
    return f"{nama} | {nim} | {kelas}"


def get_identitas_file(nama: str, nim: str, kelas: str) -> str:
    return f"{nama}_{nim}_{kelas}".replace(" ", "_")


def initial_profile(xv: np.ndarray) -> np.ndarray:
    xv = np.asarray(xv, dtype=float)
    return np.where(xv <= 0.5, 2 * xv, 2 * (1 - xv))


def explicit_diffusion(
    dx: float,
    dt: float,
    Ad: float = 1.0,
    t_end: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    lam    = Ad * dt / dx**2
    xv     = np.arange(0, 1 + dx / 2, dx)
    nsteps = int(round(t_end / dt))
    t      = np.arange(nsteps + 1) * dt

    F      = np.zeros((nsteps + 1, len(xv)))
    F[0]   = initial_profile(xv)
    F[:, 0] = F[:, -1] = 0.0

    for n in range(nsteps):
        F[n + 1, 1:-1] = F[n, 1:-1] + lam * (
            F[n, :-2] - 2 * F[n, 1:-1] + F[n, 2:]
        )
        F[n + 1, 0] = F[n + 1, -1] = 0.0

    return xv, t, F, lam


def analytic_solution(xv: np.ndarray, t: float, terms: int = 4000) -> np.ndarray:
    xv  = np.asarray(xv, dtype=float)
    n   = np.arange(1, terms + 1, dtype=float)[:, None]
    bn  = (8 / (np.pi**2 * n**2)) * np.sin(n * np.pi / 2)
    exp = np.exp(-(n * np.pi) ** 2 * float(t))
    return (bn * exp * np.sin(n * np.pi * xv)).sum(axis=0)


def save_tabel_lengkap(
    xv: np.ndarray,
    tv: np.ndarray,
    F: np.ndarray,
    name: str,
    id_file: str,
) -> str:
    df = pd.DataFrame(F, columns=[f"x={v:.1f}" for v in xv])
    df.insert(0, "t", tv)
    df.insert(0, "n", np.arange(len(tv)))

    path = f"{name.lower()}_tabel_lengkap_{id_file}.csv"
    df.to_csv(path, index=False)
    return path, df


def save_perbandingan(
    tv: np.ndarray,
    numerical: np.ndarray,
    analytic: np.ndarray,
    name: str,
    id_file: str,
) -> tuple[str, pd.DataFrame]:
    abs_err = np.abs(numerical - analytic)
    pct_err = np.where(
        np.abs(analytic) > 1e-14,
        abs_err / np.abs(analytic) * 100,
        np.nan,
    )
    cmp_df = pd.DataFrame({
        "n":             np.arange(len(tv)),
        "t":             tv,
        "numerik_x0.3":  numerical,
        "analitik_x0.3": analytic,
        "selisih_abs":   abs_err,
        "persen_error":  pct_err,
    })
    path = f"{name.lower()}_perbandingan_analitik_{id_file}.csv"
    cmp_df.to_csv(path, index=False)
    return path, cmp_df


def plot_profil_kasus(
    xv: np.ndarray,
    tv: np.ndarray,
    F: np.ndarray,
    lam: float,
    dt: float,
    dx: float,
    name: str,
    id_judul: str,
    id_file: str,
    chosen_times: list[float],
) -> str:
    chosen_idx = sorted(set(
        int(round(min(tt, tv[-1]) / dt)) for tt in chosen_times
    ))

    plt.figure(figsize=(8, 5))
    for idx in chosen_idx:
        plt.plot(xv, F[idx], marker="o", linewidth=1,
                 label=f"Numerik t={tv[idx]:.3f}")
    plt.plot(
        xv, analytic_solution(xv, tv[-1]),
        linestyle="--", linewidth=2,
        label=f"Analitik t={tv[-1]:.3f}",
    )

    label_kasus = name.replace("_", " ")
    plt.xlabel("x")
    plt.ylabel("F(x,t)")
    plt.title(f"{id_judul}\n{label_kasus} | dx={dx}, dt={dt}, λ={lam:.3f}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = f"{name.lower()}_grafik_{id_file}.png"
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()
    return path


def plot_perbandingan_semua(
    cases: dict[str, float],
    dx: float,
    Ad: float,
    t_end: float,
    x_compare: float,
    id_judul: str,
    id_file: str,
) -> str:
    plt.figure(figsize=(8, 5))

    for name, dt in cases.items():
        xv, tv, F, _ = explicit_diffusion(dx, dt, Ad=Ad, t_end=t_end)
        ix = int(round(x_compare / dx))
        plt.plot(tv, F[:, ix], marker="o", linewidth=1,
                 label=f"{name} numerik")

    t_dense        = np.linspace(0, t_end, 300)
    analytic_dense = np.array([
        analytic_solution(np.array([x_compare]), t)[0] for t in t_dense
    ])
    plt.plot(t_dense, analytic_dense, linewidth=2,
             label=f"Analitik x={x_compare}")

    plt.xlabel("t")
    plt.ylabel(f"F({x_compare},t)")
    plt.title(
        f"{id_judul}\n"
        f"Perbandingan Numerik dan Analitik di x={x_compare} (Semua Kasus)"
    )
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    path = f"perbandingan_x{str(x_compare).replace('.','')}_semua_kasus_{id_file}.png"
    plt.savefig(path, dpi=200)
    plt.show()
    plt.close()
    return path


def print_tabel(
    name: str,
    nama: str,
    nim: str,
    kelas: str,
    path_csv: str,
    df: pd.DataFrame,
) -> None:
    sep = "=" * 100
    print(f"\n{sep}")
    print(f"TABEL LENGKAP {name}  —  {nama} | {nim} | {kelas}")
    print(f"File CSV: {path_csv}")
    print(sep)
    print(df.to_string(index=False))


def print_perbandingan(
    name: str,
    nama: str,
    nim: str,
    kelas: str,
    x_compare: float,
    path_csv: str,
    cmp_df: pd.DataFrame,
) -> None:
    sep = "-" * 100
    print(f"\n{sep}")
    print(f"PERBANDINGAN NUMERIK DAN ANALITIK {name} DI x={x_compare}  —  {nama} | {nim} | {kelas}")
    print(f"File CSV: {path_csv}")
    print(sep)
    print(cmp_df.to_string(index=False))


def run_semua_kasus(
    cases: dict[str, float],
    dx: float,
    Ad: float,
    t_end: float,
    x_compare: float,
    chosen_times: list[float],
    nama: str,
    nim: str,
    kelas: str,
) -> None:
    id_judul = get_identitas_judul(nama, nim, kelas)
    id_file  = get_identitas_file(nama, nim, kelas)

    for name, dt in cases.items():
        xv, tv, F, lam = explicit_diffusion(dx, dt, Ad=Ad, t_end=t_end)

        path_tabel, df     = save_tabel_lengkap(xv, tv, F, name, id_file)
        ix                 = int(round(x_compare / dx))
        numerical          = F[:, ix]
        analytic           = np.array([analytic_solution(np.array([x_compare]), t)[0] for t in tv])
        path_cmp,  cmp_df  = save_perbandingan(tv, numerical, analytic, name, id_file)

        plot_profil_kasus(xv, tv, F, lam, dt, dx, name,
                          id_judul, id_file, chosen_times)

        print_tabel(name, nama, nim, kelas, path_tabel, df)
        print_perbandingan(name, nama, nim, kelas, x_compare, path_cmp, cmp_df)

    plot_perbandingan_semua(cases, dx, Ad, t_end, x_compare,
                            id_judul, id_file)

    print("\n" + "=" * 100)
    print("SELESAI — Semua grafik dan file CSV telah disimpan.")
    print("=" * 100)


if __name__ == "__main__":
    run_semua_kasus(
        cases        = CASES,
        dx           = DX,
        Ad           = AD,
        t_end        = T_END,
        x_compare    = X_COMPARE,
        chosen_times = CHOSEN_TIMES,
        nama         = NAMA,
        nim          = NIM,
        kelas        = KELAS,
    )
