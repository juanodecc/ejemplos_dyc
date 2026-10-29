"""
Péndulo de Furuta con stepper comandado en aceleración
Modelo no lineal con python-control (>= 0.10, ct.nlsys)

El modelo reproduce lo que hace el firmware ControlAceleracion_arduino.ino:

  * La PC manda cada Ts una aceleración ak en [rev/s^2] (texto terminado en '\\n').
  * El Arduino responde ANTES de aplicar ak: posición del stepper [pasos] y del
    encoder [cuentas]. Es decir, y_k = h(x_k).
  * Integra la velocidad con Ts fijo en el código (Ts_ard = 0.02 s):
        vk = vk_1 + ak*Ts_ard            [rev/s]
    y la deja CONSTANTE hasta el próximo mensaje (setSpeed + runSpeed).
    => la velocidad del brazo es una escalera: el péndulo recibe un impulso
       de aceleración en cada muestra, no una aceleración continua.
  * vk NO se satura: la velocidad real la limita el stepper / la tasa de
    loop(), pero el integrador sigue creciendo (windup).
  * Límite de ángulo: si |Pos| > 2500 pasos y vk empuja hacia afuera,
    vk = 0 (parada brusca, otro impulso sobre el péndulo).
  * ak == -999 resetea la velocidad a 0.
  * No hay saturación de aceleración en el firmware (setAcceleration no
    actúa con runSpeed); la única es la que aplique la PC (±255 en el swing-up).

Estados  x = [theta, v, phi, dphi]
    theta : posición del brazo (stepper)                      [rad]
    v     : velocidad COMANDADA del brazo (integrador vk)    [rad/s]
            la velocidad real es omega = clip(v, -w_max, w_max)
    phi   : ángulo del péndulo (0 = colgado, pi = invertido)  [rad]
    dphi  : velocidad del péndulo                             [rad/s]

Entrada  u = ak, aceleración enviada por serie               [rev/s^2]
Salidas  y = [theta, phi] cuantizadas como las lee el Arduino [rad]
    theta = Pos*2*pi/6400      phi = encoder*2*pi/1600

Péndulo (entre muestras omega es constante):
    phi_ddot = -2*xi*wn*dphi - wn^2*sin(phi) + kc*omega^2*sin(phi)*cos(phi)
y en cada muestra, por el salto de velocidad del brazo:
    dphi+ = dphi - K*cos(phi)*(omega+ - omega-)

Ojo con el lazo de control (furuta_swingup.ipynb): u_k se calcula con la
lectura recibida al mandar u_{k-1}, o sea hay una muestra de retardo extra
que hay que sumar al modelo al diseñar el controlador.
"""
import numpy as np
import matplotlib.pyplot as plt
import control as ct

# ---------------------------------------------------------------- parámetros
SpR = 6400              # [pasos/rev] microstepping del driver (#define SpR)
ENC_CPR = 1600          # [cuentas/rev] encoder x4 (PosPend*pi/800 en el swing-up)
POS_LIM = 2500          # [pasos] límite de ángulo del firmware
Ts = 0.02               # [s] período de muestreo del lazo en la PC

P = dict(
    wn=10.96,                       # [rad/s] frecuencia natural del péndulo (identificada)
    xi=0.008,                       # [-]     amortiguamiento
    K=12.1,                         # [1/rad*rad] acople aceleración brazo [rad/s^2] -> péndulo
    Ts_ard=0.02,                    # [s] Ts fijo en el firmware para integrar vk
    w_max=4000 / SpR * 2 * np.pi,   # [rad/s] vel. máx. real (~4000 pasos/s con AccelStepper
                                    #         en un Uno; MEDIR, setMaxSpeed no es el límite)
    th_lim=POS_LIM / SpR * 2 * np.pi,  # [rad] límite de ángulo (2500 pasos = 140.6°)
    kc=0.0,                         # término centrífugo (0 = modelo identificado; ~1 = masa puntual ideal)
    step_angle=2 * np.pi / SpR,     # [rad] cuantización de theta medido; 0 = sin cuantizar
    enc_res=2 * np.pi / ENC_CPR,    # [rad] resolución del encoder del péndulo; 0 = ideal
    n_sub=20,                       # subpasos RK4 por período (modelo discreto)
)


def _q(x, res):
    return x if res <= 0 else res * np.round(x / res)


def firmware_velocity(theta, v, u, p):
    """Nueva velocidad comandada vk [rad/s] tal como la calcula el Arduino."""
    if u == -999:
        return 0.0
    v_new = v + 2 * np.pi * u * p["Ts_ard"]
    if (theta > p["th_lim"] and v_new > 0) or (theta < -p["th_lim"] and v_new < 0):
        v_new = 0.0
    return v_new


def _pend_accel(phi, dphi, omega, p):
    return (-2 * p["xi"] * p["wn"] * dphi
            - p["wn"] ** 2 * np.sin(phi)
            + p["kc"] * omega ** 2 * np.sin(phi) * np.cos(phi))


# ------------------------------------------------- modelo discreto (fiel al firmware)
def furuta_update_d(t, x, u, params):
    p = {**P, **params}
    theta, v, phi, dphi = x
    w_old = np.clip(v, -p["w_max"], p["w_max"])
    v = firmware_velocity(theta, v, u[0], p)
    w = np.clip(v, -p["w_max"], p["w_max"])

    # impulso por el salto de velocidad del brazo
    dphi = dphi - p["K"] * np.cos(phi) * (w - w_old)

    # brazo a velocidad constante, péndulo libre durante Ts (RK4)
    h = Ts / p["n_sub"]
    for _ in range(int(p["n_sub"])):
        k1p, k1v = dphi, _pend_accel(phi, dphi, w, p)
        k2p, k2v = dphi + h / 2 * k1v, _pend_accel(phi + h / 2 * k1p, dphi + h / 2 * k1v, w, p)
        k3p, k3v = dphi + h / 2 * k2v, _pend_accel(phi + h / 2 * k2p, dphi + h / 2 * k2v, w, p)
        k4p, k4v = dphi + h * k3v, _pend_accel(phi + h * k3p, dphi + h * k3v, w, p)
        phi = phi + h / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        dphi = dphi + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)

    return [theta + w * Ts, v, phi, dphi]


def furuta_output(t, x, u, params):
    p = {**P, **params}
    return [_q(x[0], p["step_angle"]), _q(x[2], p["enc_res"])]


furuta = ct.nlsys(
    furuta_update_d, furuta_output, name="furuta", dt=Ts,
    inputs=["u"], outputs=["theta", "phi"],
    states=["theta", "v", "phi", "dphi"],
    params=P,
)


# ------------------------------- aproximación continua (Ts -> 0), para análisis
def furuta_update_c(t, x, u, params):
    """Sin límite de ángulo (es un reset de estado) y con a = 2*pi*u."""
    p = {**P, **params}
    theta, v, phi, dphi = x
    dv = 2 * np.pi * u[0]
    a = dv if abs(v) < p["w_max"] else 0.0      # windup: v sigue, omega no
    omega = np.clip(v, -p["w_max"], p["w_max"])
    ddphi = _pend_accel(phi, dphi, omega, p) - p["K"] * a * np.cos(phi)
    return [omega, dv, dphi, ddphi]


furuta_c = ct.nlsys(
    furuta_update_c, furuta_output, name="furuta_c",
    inputs=["u"], outputs=["theta", "phi"],
    states=["theta", "v", "phi", "dphi"],
    params=P,
)


# ------------------------ modelo lineal continuo identificado (péndulo colgado)
#   theta(s)/u(s) = kstepper / s^2
#   phi(s)/u(s)   = k / (s^2 + 2*xi*wn*s + wn^2)
# u y las unidades de theta, phi son las usadas en la identificación.
P_lin = dict(kstepper=7221.0, k=12.1, xi=0.008, wn=10.956)


def furuta_update_lin(t, x, u, params):
    p = {**P_lin, **params}
    theta, dtheta, phi, dphi = x
    ddtheta = p["kstepper"] * u[0]
    ddphi = -2 * p["xi"] * p["wn"] * dphi - p["wn"] ** 2 * phi + p["k"] * u[0]
    return [dtheta, ddtheta, dphi, ddphi]


def furuta_output_lin(t, x, u, params):
    return [x[0], x[2]]


furuta_lin = ct.nlsys(
    furuta_update_lin, furuta_output_lin, name="furuta_lin",
    inputs=["u"], outputs=["theta", "phi"],
    states=["theta", "dtheta", "phi", "dphi"],
    params=P_lin,
)


def simulate(u_fun, T, x0, **kw):
    """Simula el modelo discreto: u_fun(t) en [rev/s^2], muestreado cada Ts."""
    t = np.arange(0, T, Ts)
    u = np.array([u_fun(ti) for ti in t])
    return ct.input_output_response(furuta, t, u, x0, return_x=True, **kw)


# -------------------------------------------------------------- linealización
def linearize_inverted(discrete=True):
    """Linealiza en phi = pi (péndulo arriba), lejos de las saturaciones."""
    sys = furuta if discrete else furuta_c
    return ct.linearize(sys, [0, 0, np.pi, 0], [0], params=dict(step_angle=0, enc_res=0))


if __name__ == "__main__":
    # 1) Demo: saturación de velocidad con windup y límite de ángulo
    demo = dict(w_max=2.0)

    def u_cmd(t):
        if 0.1 < t < 1.1:   return 0.5      # acelera: satura omega, v sigue creciendo
        if 1.1 <= t < 2.1:  return -0.5     # frena: omega no baja hasta descargar v
        return 0.0

    r = simulate(u_cmd, 6.0, [0, 0, 0, 0], params=demo)
    t, th, phi = r.time, r.outputs[0], r.outputs[1]
    v = r.states[1]
    omega = np.clip(v, -demo["w_max"], demo["w_max"])

    fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
    ax[0].step(t, [u_cmd(ti) for ti in t], where="post")
    ax[0].set_ylabel("u [rev/s²]")
    ax[1].step(t, v, where="post", label="v comandada")
    ax[1].step(t, omega, where="post", label="ω real")
    ax[1].axhline(demo["w_max"], ls=":", c="k"); ax[1].axhline(-demo["w_max"], ls=":", c="k")
    ax[1].set_ylabel("brazo [rad/s]"); ax[1].legend()
    ax[2].step(t, th, where="post")
    ax[2].axhline(P["th_lim"], ls=":", c="k"); ax[2].axhline(-P["th_lim"], ls=":", c="k")
    ax[2].set_ylabel("θ [rad]")
    ax[3].step(t, phi, where="post"); ax[3].set_ylabel("φ [rad]"); ax[3].set_xlabel("t [s]")
    for a in ax: a.grid(True)
    plt.tight_layout(); plt.savefig("furuta_saturacion.png", dpi=120)

    # 2) Linealización en la vertical invertida
    for name, lin in [("discreto", linearize_inverted(True)), ("continuo", linearize_inverted(False))]:
        print(f"--- {name}")
        print("A =\n", np.round(lin.A, 4))
        print("B =\n", np.round(lin.B, 4))
        print("polos:", np.round(np.linalg.eigvals(lin.A), 4))
        print("controlable:", np.linalg.matrix_rank(ct.ctrb(lin.A, lin.B)) == 4)
