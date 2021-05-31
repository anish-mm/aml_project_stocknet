import torch
from torch.optim import adamw

max_n_days = 5
max_n_msgs = 30
max_n_words = 40

batch_size = 7
vocab_size = 15
word_embed_size = 2
mel_h_size = 20
msg_embed_size = mel_h_size
price_size = 3

dropout_mel_in = 0.5
dropout_mel = 0.3
mel_cell_type = "gru" # "ln-lstm", "basic"

dropout_ce = 0.2

variant_type="hedge"
daily_att = 'y'
vmd_rec="zh"
dropout_vmd_in=0.2
dropout_vmd=0.2
vmd_cell_type="gru"
h_size=30
g_size=10
y_size=2
z_size=h_size

kl_lambda_anneal_rate= 0.005  # 0.005, 0.002, 0.001, 0.0005
kl_lambda_start_step= 0
alpha= 0.5
use_constant_kl_lambda= 0
constant_kl_lambda= 0.1

def test_batch_compacter():
    from model import BatchCompacter

    compacter = BatchCompacter()
    prices = torch.randn(batch_size, max_n_days, price_size)
    words = torch.randint(0, vocab_size, (batch_size, max_n_days, max_n_msgs, max_n_words))
    n_days = torch.randint(1, max_n_days, (batch_size,))
    n_msgs = torch.randint(1, max_n_msgs, (batch_size, max_n_days,))
    n_words = torch.randint(1, max_n_words, (batch_size, max_n_days, max_n_msgs,))
    c_out = compacter(prices, words, n_days, n_msgs, n_words)

    # some compacting is certain.

    max_valid_n_days = n_days.max()
    max_valid_n_msgs = n_msgs[:, :max_valid_n_days].max()
    max_valid_n_words = n_words[ :, :max_valid_n_days, :max_valid_n_msgs].max()
    
    print(f"""
    Testing BatchCompacter
    ----------------------
    max_n_days = {max_n_days},
    max_n_msgs = {max_n_msgs},
    max_n_words = {max_n_words},

    max_valid_n_days = {max_valid_n_days},
    max_valid_n_msgs = {max_valid_n_msgs},
    max_valid_n_words = {max_valid_n_words},

    words.shape = {words.shape},
    """)
    print(f"n_days = {n_days}")

    checks = [
        ("price", c_out["prices"].shape, (batch_size, max_valid_n_days, price_size)),
        ("max_n_msgs", c_out["max_n_msgs"], max_valid_n_msgs),
        ("words", c_out["words"].shape, (batch_size, max_valid_n_days, max_valid_n_msgs, max_valid_n_words)),
        ("n_days", c_out["n_days"], n_days),
        ("max_n_days", c_out["max_n_days"], max_valid_n_days),
        ("n_msgs", c_out["n_msgs"], n_msgs[:, :max_valid_n_days]),
        ("n_words", c_out["n_words"], n_words[:, :max_valid_n_days, :max_valid_n_msgs]),
        ("max_n_words", c_out["max_n_words"], max_valid_n_words),
    ]
    
    for name, vi, vj in checks:
        print(f"""
        name : {name}
        vi   : {vi}
        vj   : {vj}
        """)
        cond = (vi == vj)
        try:
            cond = cond.all()
        except:
            continue

        assert cond, f"Assertion failed for {name}!"
    
    print("Test passed.")

def test_word_embedder():
    from model import WordEmbedder

    prices = torch.randn(batch_size, max_n_days, price_size)
    words = torch.randint(0, vocab_size, (batch_size, max_n_days, max_n_msgs, max_n_words))
    n_days = torch.randint(1, max_n_days, (batch_size,))
    n_msgs = torch.randint(1, max_n_msgs, (batch_size, max_n_days,))
    n_words = torch.randint(1, max_n_words, (batch_size, max_n_days, max_n_msgs,))
    word_table = torch.randn(vocab_size, word_embed_size)

    we = WordEmbedder(word_table)
    word_embed = we(words)

    print(f"""
    Testing WordEmbedder
    --------------------
    word_embed.shape : {word_embed.shape}
    expected shape   : {(batch_size, max_n_days, max_n_msgs, max_n_words, word_embed_size)}
    """)

    assert word_embed.shape == (batch_size, max_n_days, max_n_msgs, max_n_words, word_embed_size),\
        "shape mismatch."
    
    for i in range(batch_size):
        for j in range(max_n_days):
            for k in range(max_n_msgs):
                for l in range(max_n_words):
                    assert torch.equal(word_embed[i, j, k, l], word_table[words[i, j, k, l]]), \
                        "table lookup has errors."

    print("Test passed.")

def test_message_embedder():
    from model import MessageEmbedder
    from model import WordEmbedder
    from model import BatchCompacter

    prices = torch.randn(batch_size, max_n_days, price_size)
    words = torch.randint(0, vocab_size, (batch_size, max_n_days, max_n_msgs, max_n_words))
    n_days = torch.randint(1, max_n_days, (batch_size,))
    n_msgs = torch.randint(1, max_n_msgs, (batch_size, max_n_days,))
    n_words = torch.randint(1, max_n_words, (batch_size, max_n_days, max_n_msgs,))


    compacter = BatchCompacter()
    c_out = compacter(prices, words, n_days, n_msgs, n_words)

    word_table = torch.randn(vocab_size, word_embed_size)
    we = WordEmbedder(word_table)
    word_embed = we(c_out["words"])

    for mel_cell_type in ["gru", "ln-lstm", "basic"]:

        me = MessageEmbedder(dropout_mel_in=dropout_mel_in, dropout_mel=dropout_mel,
            word_embed_size=word_embed_size, mel_h_size=mel_h_size,
            mel_cell_type=mel_cell_type,
        )
        msg_embed = me(word_embed, c_out["n_words"])
        msg_embed_shape = (batch_size, c_out["max_n_days"], c_out["max_n_msgs"], msg_embed_size)
        
        print(f"""
        Testing MessageEmbedder
        -----------------------
        expected shape   : {msg_embed_shape}
        returned shape   : {msg_embed.shape}
        """)

        assert msg_embed.shape == msg_embed_shape, f"shape mismatch"
    print("Test passed.")

def test_corpus_embedder():
    from model import MessageEmbedder
    from model import WordEmbedder
    from model import BatchCompacter
    from model import CorpusEmbedder

    prices = torch.randn(batch_size, max_n_days, price_size)
    words = torch.randint(0, vocab_size, (batch_size, max_n_days, max_n_msgs, max_n_words))
    n_days = torch.randint(1, max_n_days, (batch_size,))
    n_msgs = torch.randint(1, max_n_msgs, (batch_size, max_n_days,))
    n_words = torch.randint(1, max_n_words, (batch_size, max_n_days, max_n_msgs,))


    compacter = BatchCompacter()
    c_out = compacter(prices, words, n_days, n_msgs, n_words)

    word_table = torch.randn(vocab_size, word_embed_size)
    we = WordEmbedder(word_table)
    word_embed = we(c_out["words"])

    me = MessageEmbedder(dropout_mel_in=dropout_mel_in, dropout_mel=dropout_mel,
        word_embed_size=word_embed_size, mel_h_size=mel_h_size,
        mel_cell_type=mel_cell_type,
    )
    msg_embed = me(word_embed, c_out["n_words"])

    ce = CorpusEmbedder(mel_h_size=mel_h_size, dropout_ce=dropout_ce)
    corpus_embed = ce(msg_embed, c_out["n_msgs"])

    corpus_embed_shape = (batch_size, c_out["max_n_days"], mel_h_size)
    print(f"""
    Testing CorpusEmbedder
    ----------------------
    expected shape : {corpus_embed_shape}
    returned shape : {corpus_embed.shape}
    """)

    assert corpus_embed_shape == corpus_embed.shape, "shape mismatch."

    print("Test passed.")

def test_mie():
    from model import MIE

    prices = torch.randn(batch_size, max_n_days, price_size)
    words = torch.randint(0, vocab_size, (batch_size, max_n_days, max_n_msgs, max_n_words))
    n_days = torch.randint(1, max_n_days, (batch_size,))
    n_msgs = torch.randint(1, max_n_msgs, (batch_size, max_n_days,))
    n_words = torch.randint(1, max_n_words, (batch_size, max_n_days, max_n_msgs,))
    word_table = torch.randn(vocab_size, word_embed_size)
    max_valid_n_days = n_days.max()
    checks = {
        "hedge": (batch_size, max_valid_n_days, price_size + mel_h_size),
        "tech":(batch_size, max_valid_n_days, price_size),
        "fund": (batch_size, max_valid_n_days, mel_h_size),
        "discriminative": (batch_size, max_valid_n_days, price_size + mel_h_size),
    }

    print("""
    Testing mie
    -----------
    """)
    for variant_type in ["hedge", "tech", "fund", "discriminative"]:

        mie = MIE(
            word_table_init=word_table, 
            dropout_mel_in=dropout_mel_in, dropout_mel=dropout_mel,
            word_embed_size=word_embed_size, mel_h_size=mel_h_size,
            mel_cell_type=mel_cell_type,
            price_size=price_size,
            variant_type=variant_type, dropout_ce=dropout_ce,
        )
        x, mnd = mie(
            prices=prices, words=words, n_days=n_days,
            n_msgs=n_msgs, n_words=n_words
        )

        print(f"""
        variant        : {variant_type}
        expected shape : {checks[variant_type]}
        returned shape : {x.shape}

        """)

        assert x.shape == checks[variant_type], "shape mismatch"
    
    print("Test passed.")

def test_vmd_zh_rec():
    from model import VMDWithZHRec
    from model import MIE

    prices = torch.randn(batch_size, max_n_days, price_size)
    words = torch.randint(0, vocab_size, (batch_size, max_n_days, max_n_msgs, max_n_words))
    n_days = torch.randint(1, max_n_days, (batch_size,))
    n_msgs = torch.randint(1, max_n_msgs, (batch_size, max_n_days,))
    n_words = torch.randint(1, max_n_words, (batch_size, max_n_days, max_n_msgs,))
    word_table = torch.randn(vocab_size, word_embed_size)
    max_valid_n_days = n_days.max()
    y_true = torch.nn.functional.one_hot(torch.randint(0, 2, (batch_size, max_valid_n_days)))

    mie = MIE(
        word_table_init=word_table, 
        dropout_mel_in=dropout_mel_in, dropout_mel=dropout_mel,
        word_embed_size=word_embed_size, mel_h_size=mel_h_size,
        mel_cell_type=mel_cell_type,
        price_size=price_size,
        variant_type=variant_type, dropout_ce=dropout_ce,
    )
    x, mnd = mie(
        prices=prices, words=words, n_days=n_days,
        n_msgs=n_msgs, n_words=n_words
    )
    if variant_type == "tech":
        x_size=price_size
    elif variant_type == "fund":
        x_size=mel_h_size
    else:
        x_size = price_size + mel_h_size

    print("""
    Testing VMDWithZHRec
    --------------------
    """)
    for daily_att in ["y", "g"]:
        vmd = VMDWithZHRec(
            dropout_vmd_in=dropout_vmd_in, dropout_vmd=dropout_vmd,
            vmd_cell_type=vmd_cell_type, h_size=h_size, x_size=x_size,
            y_size=y_size, z_size=z_size, g_size=g_size, 
            daily_att=daily_att,
            vmd_rec=vmd_rec, variant_type=variant_type,
        )

        vmd_out = vmd(
            x, y_true, n_days,
        )
        
        print(f"""
        daily_att     : {daily_att}
        vmd_out       : {vmd_out}
        """)
        break

def test_stocknet():
    from model import VMD
    from model import MIE
    from model import Stocknet
    from torch.optim import Adam

    prices = torch.randn(batch_size, max_n_days, price_size)
    words = torch.randint(0, vocab_size, (batch_size, max_n_days, max_n_msgs, max_n_words))
    n_days = torch.randint(2, max_n_days, (batch_size,))
    n_msgs = torch.randint(1, max_n_msgs, (batch_size, max_n_days,))
    n_words = torch.randint(1, max_n_words, (batch_size, max_n_days, max_n_msgs,))
    word_table = torch.randn(vocab_size, word_embed_size)
    max_valid_n_days = n_days.max()
    y_true = torch.nn.functional.one_hot(torch.randint(0, 2, (batch_size, max_valid_n_days)))

    net = Stocknet(
        word_table_init=word_table, 
        dropout_mel_in=dropout_mel_in, dropout_mel=dropout_mel,
        word_embed_size=word_embed_size, mel_h_size=mel_h_size,
        mel_cell_type=mel_cell_type,
        price_size=price_size,
        variant_type=variant_type, dropout_ce=dropout_ce,
        dropout_vmd_in=dropout_vmd_in, dropout_vmd=dropout_vmd,
        vmd_cell_type=vmd_cell_type, h_size=h_size,
        y_size=y_size, z_size=z_size, g_size=g_size, 
        daily_att=daily_att,
        vmd_rec=vmd_rec,
        alpha=alpha,
        kl_lambda_anneal_rate=kl_lambda_anneal_rate,
        kl_lambda_start_step=kl_lambda_start_step,
        constant_kl_lambda=constant_kl_lambda, 
        use_constant_kl_lambda=use_constant_kl_lambda,
    )

    opt = Adam(net.parameters(), lr=0.001)

    net.train()
    # net.eval()

    sample_index = torch.arange(batch_size).unsqueeze(-1)
    day_index = torch.reshape(n_days - 1, (batch_size, 1))
    y_true_T = y_true[sample_index, day_index]

    for i in range(30):
        y_T, loss = net(
            prices=prices, words=words, n_days=n_days,
            n_msgs=n_msgs, n_words=n_words,
            y_true=y_true, global_step=0
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

        print(f"""
        iter  : {i + 1}
        y_T   : {y_T}
        y_true: {y_true_T}
        loss  : {loss}
        """)

    
for key, val in thedict.items():
    try:
        print(key, val.shape)
    except:
        try:
            print(key, len(val))
        except:
            continue