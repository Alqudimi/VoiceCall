# --------------------------------------------------------------------------- #
# Adapted from the official RVC repository:
#   https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
# (c) RVC-Project contributors — MIT licence.
# Adaptations performed by Iraqi Voice Studio (documented in
# docs/ARCHITECTURE.md and engine/rvc/ATTRIBUTION.md):
#   * imports rewritten to the ``rvc_platform`` package layout
#   * i18n / gradio / webui couplings removed
#   * asset paths routed through rvc_platform.infrastructure.paths
#   * no algorithmic changes to numerics or model structure
# --------------------------------------------------------------------------- #
import torch


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach()
            gl = gl.float()
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1 - dr) ** 2)
        g_loss = torch.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l_gen = torch.mean((1 - dg) ** 2)
        gen_losses.append(l_gen)
        loss += l_gen

    return loss, gen_losses


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l_kl = kl / torch.sum(z_mask)
    return l_kl
