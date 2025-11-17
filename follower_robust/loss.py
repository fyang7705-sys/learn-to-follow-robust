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

# metric loss in FOCAL
def metric_loss(z, tasks, beta = 1.0, n = 2,epsilon=1e-6):
    # z shape is (task, dim)
    pos_z_loss = 0.
    neg_z_loss = 0.
    pos_cnt = 0
    neg_cnt = 0

    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            # positive pair
            if tasks[i] == tasks[j]:
                # pos_z_loss += torch.sqrt(torch.mean((z[i] - z[j]) ** 2) + epsilon)
                pos_z_loss += torch.sum((z[i] - z[j]) ** 2)
                pos_cnt += 1
            else:
                # neg_z_loss += 1/(torch.mean((z[i] - z[j]) ** 2) + epsilon * 100)
                dist = torch.sqrt(torch.sum((z[i] - z[j]) ** 2) + epsilon)
                neg_z_loss += 1 / (dist ** n + epsilon)
                neg_cnt += 1
    pos_z_loss = pos_z_loss / (pos_cnt + epsilon)
    neg_z_loss = neg_z_loss / (neg_cnt + epsilon)
    #print(pos_z_loss, pos_cnt, neg_z_loss, neg_cnt)
    return pos_z_loss, pos_cnt, neg_z_loss, neg_cnt, pos_z_loss + beta * neg_z_loss

def metric_loss_fast(z, tasks, beta=1.0, n=2, epsilon=1e-6):
    N = z.shape[0]

    # ----- pairwise squared L2 distance matrix -----
    diff = z.unsqueeze(1) - z.unsqueeze(0) # (N, N, dim)
    dist2 = torch.sum(diff * diff, dim=2) # ||z_i - z_j||^2
    dist = torch.sqrt(dist2 + epsilon) # ||z_i - z_j||

    # ----- positive mask -----
    task_eq = (tasks.unsqueeze(0) == tasks.unsqueeze(1)) # (N, N)
    pos_mask = torch.triu(task_eq, diagonal=1)
    neg_mask = torch.triu(~task_eq, diagonal=1)

    # ----- positive loss -----
    pos_dist2 = dist2[pos_mask]
    pos_loss = pos_dist2.mean() if pos_dist2.numel() > 0 else torch.tensor(0., device=z.device)
    pos_cnt = pos_dist2.numel()

    # ----- negative loss -----
    neg_dist = dist[neg_mask]
    neg_loss = (1.0 / (neg_dist**n + epsilon)).mean() if neg_dist.numel() > 0 else torch.tensor(0., device=z.device)
    neg_cnt = neg_dist.numel()

    # ----- total -----
    loss = pos_loss + beta * neg_loss

    return pos_loss, pos_cnt, neg_loss, neg_cnt, loss


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