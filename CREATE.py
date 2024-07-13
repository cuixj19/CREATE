#!/usr/bin/env python
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cis-Regulatory Elements identificAtion via discreTe Embedding')
    
    parser.add_argument('--data_path', '-d', type=str, default='./example/')
    parser.add_argument('--num_class', '-n', type=int, default=5)
    parser.add_argument('--multi', type=str, default=['seq','open','loop'])
    parser.add_argument('--test_aug', type=int, default=1)
    parser.add_argument('--train_aug', type=int, default=[1,1,1,1,1])
    parser.add_argument('--stride', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=None)
    
    parser.add_argument('--enc_dims', type=int, default=[512, 384, 128])
    parser.add_argument('--dec_dims', type=int, default=[200, 200])
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_embed', type=int, default=200)
    parser.add_argument('--split', type=int, default=16)
    parser.add_argument('--no_ema', action='store_false')
    parser.add_argument('--e_loss_weight', type=float, default=0.25)
    parser.add_argument('--mu', type=float, default=0.01)
    
    parser.add_argument('--open_loss_weight', type=float, default=0.01)
    parser.add_argument('--loop_loss_weight', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--max_epoch', type=int, default=300)
    parser.add_argument('--pre_epoch', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--outdir', '-o', type=str, default='./output/')

    args = parser.parse_args()
    
    import create
    create.main.CREATE(
        data_path=args.data_path, 
        num_class=args.num_class, 
        multi=args.multi, 
        test_aug=args.test_aug, 
        train_aug=args.train_aug, 
        stride=args.stride, 
        batch_size=args.batch_size, 
        enc_dims=args.enc_dims, 
        dec_dims=args.dec_dims, 
        embed_dim=args.embed_dim, 
        n_embed=args.n_embed, 
        split=args.split, 
        ema=(not args.no_ema), 
        e_loss_weight=args.e_loss_weight, 
        mu=args.mu, 
        open_loss_weight=args.open_loss_weight, 
        loop_loss_weight=args.loop_loss_weight, 
        lr=args.lr, 
        max_epoch=args.max_epoch, 
        pre_epoch=args.pre_epoch, 
        seed=args.seed, 
        gpu=args.gpu, 
        outdir=args.outdir, 
    )
        
