import torch


def uas(predicted: torch.Tensor, gold: torch.Tensor, ignore_idx: int) -> torch.Tensor:
    non_ignore = (gold != ignore_idx).nonzero()
    count_matches = predicted[non_ignore].squeeze(0).eq(gold[non_ignore])

    return count_matches.sum() / gold[non_ignore].size()[0]


def batch_uas(predicted: torch.Tensor, gold: torch.Tensor, ignore_idx: int) -> torch.Tensor:
    all_uas = list()
    for pred, go in zip(predicted, gold):
        all_uas.append(uas(pred, go, ignore_idx))

    return torch.mean(torch.tensor(all_uas))


def las(pred_heads: torch.Tensor, gold_heads: torch.Tensor,
        pred_rels: torch.Tensor, gold_rels: torch.Tensor,
        ignore_idx: int) -> torch.Tensor:
    non_ignore = (gold_heads != ignore_idx)

    x = (pred_heads[non_ignore] == gold_heads[non_ignore]
         and pred_rels == gold_rels).nonzero()
    print(x.sum())

    return torch.tensor(0)


if __name__ == "__main__":
    # s = uas(torch.arange(20), torch.arange(20), ignore_idx=0)
    # print(s)

    # print(batch_uas(
    #     torch.randint(0, 20, (4, 20)),
    #     torch.randint(0, 20, (4, 20)),
    #     2
    # ))

    las(
        torch.arange(20), torch.arange(20),
        torch.arange(20), torch.arange(20),
        2
    )
