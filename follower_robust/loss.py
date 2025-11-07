import torch


def pgm_loss(student_probs, teacher_actions, eta=0.3, eps=1e-8):

    batch_loss = []
    for i in range(student_probs.size(0)):
        p = student_probs[i]
        aT = teacher_actions[i].item()
        p_aT = p[aT]
        p_rest = torch.cat([p[:aT], p[aT+1:]])
        p_rest_norm = p_rest / (1 - p_aT + eps)

        entropy = -torch.sum(p_rest_norm * torch.log(p_rest_norm + eps))
        loss_i = - (p_aT ** eta) * entropy
        batch_loss.append(loss_i)
    return torch.stack(batch_loss).mean()


def jacobian_regularization(loss_pgm, obs_tensor):

    grad = torch.autograd.grad(
        outputs=loss_pgm,
        inputs=obs_tensor,
        grad_outputs=torch.ones_like(loss_pgm),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
        allow_unused=True
    )[0]
    if grad is None:
        return torch.tensor(0.0, device=loss_pgm.device)

    B = grad.shape[0]
    per_sample_norm = grad.view(B, -1).pow(2).sum(dim=1)  # Frobenius per sample
    return per_sample_norm.mean()  # scalar