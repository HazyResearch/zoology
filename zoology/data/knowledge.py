import torch
from .utils import SyntheticData, builder_from_single


def fact_retrieval(
    input_seq_len: int = 3,
    vocab_size: int = 8_192,
    n_subjects: int = 1024, 
    p_predicates: int = 1, 
    m_objects: int = 1024,
    density: float = 1.0,
    seed: int = 0,
):
    # set seed
    torch.manual_seed(seed)

    required_vocab_size = n_subjects + p_predicates + m_objects + 1
    if vocab_size < required_vocab_size:
        raise ValueError(
            f"vocab_size must be at least {required_vocab_size} but got {vocab_size}"
        )
    
    null_token = 0
    predicates = torch.arange(1, p_predicates + 1)
    subjects = torch.arange(p_predicates + 1, p_predicates + n_subjects + 1)
    
    all_data = []
    for predicate in range(p_predicates):
        data = torch.zeros(n_subjects, 3)    
        data[:, 0] = subjects
        data[:, 1] = predicates[predicate]
        # TODO: randomly permute objects here
        data[:, 2] = torch.randperm(m_objects) + p_predicates + n_subjects + 1
        all_data.append(data.long())

    data = torch.cat(all_data, dim=0)
    
    # pad the data with null tokens to the input_seq_len
    padding = input_seq_len - data.shape[1]
    if padding > 0:
        data = torch.nn.functional.pad(data, (0, padding), "constant", 0).long()
    
    # create an x and y where the x masks out the object with 0 and the y masks out 
    # the subject and predicate with -100
    x = data.clone()
    x[:, 2] = null_token
    y = data.clone()
    y[:, 0] = -100
    y[:, 1] = -100

    return SyntheticData(
        train_inputs=x,
        train_labels=y,
        test_inputs=x,
        test_labels=y,
    )


        



# def constrained_search(
#     n_subjects: int = 1024, 
#     p_predicates: int = 1024,
#     m_objects_per_predicate: int = 32,
#     d_model: int = 16, 
# ):
#     objects = []
#     predicates = []
#     for _ in range(p_predicates):
#         mapping = torch.arange(0, m_objects_per_predicate).repeat(n_subjects // m_objects_per_predicate)
#         mapping = mapping[torch.randperm(n_subjects)]
#         embs = nn.init.normal_(torch.zeros(m_objects_per_predicate, d_model // 2))
#         predicates.append(mapping)
#         objects.append(embs[mapping])
    
#     # y = nn.init.normal_(torch.zeros(n_subjects, d_model)).to(device)
#     xs, ys = [], []
#     for i in range(p_predicates):
#         for j in range(i + 1, p_predicates):
#             for k in range(m_objects_per_predicate):
#                 for l in range(m_objects_per_predicate):
#                     xs.append(torch.concat([objects[i][k], objects[j][l]], dim=-1))
#                     ys.append(int(((predicates[i] == k) & (predicates[j] == l)).any()))
#     x = torch.stack(xs, dim=0)
#     y = torch.tensor(ys)

#     return x, y
# x, y = constrained_search(
#     n_subjects=32,
#     p_predicates=32,
#     m_objects_per_predicate=8,
#     d_model=12,
# )