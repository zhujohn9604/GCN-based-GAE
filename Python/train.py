import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import os
import pickle
import dgl
import dgl.function as fn

from collections import defaultdict
from data_year import PAPER_TYPE, SUBJECT_TYPE, load_data
from Preprocessing import AlphaNumericTextPreprocessor
from gcn import GCNSampling
from dgl.contrib.sampling.sampler import NeighborSampler
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)




# Model directory structure
MDS = {
    "embedding_file": "embedding.csv",
    "args_file": "args.txt",
    "loss_file": "losses.txt",
    "accuracy_file": "accuracy.txt",
    "temporal_embeddings_dir": "temporal_embeddings",
    "ccount_file": "cumulative_counts.csv",
    "checkpoints_dir": "checkpoints"
}


def preprocess_text(args, data):
    print("Preprocessing text...")
    preprocessor = AlphaNumericTextPreprocessor(max_features=args.max_features, lowercase=True,
                                                max_length=args.max_length,
                                                stop_words=ENGLISH_STOP_WORDS, drop_unknown=True, dtype=torch.tensor)
    titles = data.paper_features['appln_abstract'].values
    text_features = preprocessor.fit_transform(titles)
    return text_features, preprocessor


def embed_control_variate(args, data):
    num_neighbors = args.num_neighbors
    n_layers = args.n_layers
    dropout = args.dropout
    emb_size = args.embedding_dim
    n_hidden = args.n_hidden
    rep_size = args.representation_size

    device = torch.device("cuda") if args.use_cuda else torch.device("cpu")
    globals_device = device if not args.globals_on_cpu else torch.device("cpu")

    # Preparing lists of vertices
    subjects = data.ndata[data.ndata.type == SUBJECT_TYPE]
    subject_vs = subjects.index.values
    paper_vs = data.ndata[data.ndata.type == PAPER_TYPE].index.values
    print(subject_vs)
    print(len(subject_vs))

    print("Networkx graph")
    print("Number of nodes:", data.graph.number_of_nodes())
    print("Number of edges:", data.graph.number_of_edges())

    if args.warm_start:
        #try:
        #    with open(os.path.join(args.out, "preprocessor.pkl"), 'rb') as fh:
        #        preprocessor = pickle.load(fh)
        #    text_features = preprocessor.transform(data.paper_features['appln_abstract'].values)
        #except FileNotFoundError:
        #    print("Warning: warm start without restoring preprocessor...")
        #    print("Vocabulary will be recreated.")
        #    text_features, preprocessor = preprocess_text(args, data)
    #else:
        text_features, preprocessor = preprocess_text(args, data)

    print("Text feature dims", text_features.size())

    #with open(os.path.join(args.out, "preprocessor.pkl"), 'wb') as fh:
    #    print("Saving preprocessor to", fh.name)
    #    pickle.dump(preprocessor, fh)

    print("Creating DGL graph ...")
    g = dgl.DGLGraph(data.graph, readonly=True)
    g.set_n_initializer(dgl.init.zero_initializer)
    print("DGL Graph")
    print("Number of nodes:", g.number_of_nodes())
    print("Number of edges:", g.number_of_edges())

    # --INIT 'features'--
    print('Adding features')
    features = torch.zeros(g.number_of_nodes(), text_features.size(1),
                           device=text_features.device, dtype=text_features.dtype)
    features[paper_vs] = text_features
    g.ndata['features'] = features.to(globals_device)
    print("Feature size", features.size())

    print("Mapping subject nids to class label")
    print("Subj values", subject_vs)
    subject2classlabel = defaultdict(lambda: -1, {nid: c for c, nid in enumerate(subject_vs)})
    # all nids that are not in the subject_vs will return -1
    n_classes = len(subject2classlabel)
    targets = torch.zeros(g.number_of_nodes(), dtype=torch.int64) - 1
    for nid in subject_vs:
        targets[nid] = subject2classlabel[nid]

    targets = targets.to(device)  # on device: gpu

    print("Subject targets:", targets[subject_vs])
    print("Number of classes:", n_classes)

    g.ndata['h_{}'.format(0)] = torch.zeros(g.number_of_nodes(),
                                            emb_size,
                                            device=globals_device)
    for i in range(1, n_layers):
        g.ndata['h_{}'.format(i)] = torch.zeros(g.number_of_nodes(),
                                                n_hidden,
                                                device=globals_device)

    # penultimate skip-connection layer
    if n_layers > 1:
        g.ndata['h_{}'.format(n_layers - 1)] = torch.zeros(g.number_of_nodes(),
                                                           2 * n_hidden,
                                                           device=globals_device)

    # For two layers
    # h_0 : [N, emb_size]
    # h_1 : [N, 2*n_hidden]

    # --INIT 'norm'--
    print("Computing global norm...")
    norm = 1. / g.in_degrees().float().unsqueeze(1)
    norm[torch.isinf(norm)] = 0.
    print("Norm", norm, sep='\n')
    print("Norm size", norm.size())  # size: #nodes x 1/degree(in = out)
    g.ndata['norm'] = norm.to(globals_device)

    text_encoder = nn.Embedding(len(preprocessor.vocabulary_) + 1,
                                emb_size,
                                sparse=(not args.scale_grad_by_freq), padding_idx=0,
                                max_norm=args.max_norm,
                                scale_grad_by_freq=args.scale_grad_by_freq)

    if args.embedding_dropout:
        text_encoder = nn.Sequential(text_encoder, nn.Dropout(args.embedding_dropout))

    model = GCNSampling(emb_size,
                        n_hidden,
                        rep_size,
                        n_layers,
                        F.relu,
                        dropout)

    # Linear decoder
    decoder = nn.Linear(rep_size, n_classes, bias=args.decoder_bias)

    if args.representation_dropout \
            or args.representation_activation \
            or args.representation_layer_norm:
        pp_modules = []
        if args.representation_dropout and args.representation_layer_norm:
            print("warning: dropout and layer norm might not go well together")
        if args.representation_dropout:
            pp_modules.append(nn.Dropout(args.representation_dropout))
        if args.representation_activation:
            act = getattr(torch.nn, args.representation_activation)()
            pp_modules.append(act)
        if args.representation_layer_norm:
            pp_modules.append(nn.LayerNorm(rep_size, elementwise_affine=False))
        model = nn.Sequential(model, *pp_modules)

    if args.warm_start:
        print("Loading model checkpoint from", args.out)
        print('Loading pretrained word embeddings...')
        pretrained_wordemb = np.load(args.graphdir + 'pretrained_wordvectors.npy')
        text_encoder.weight.data.copy_(torch.from_numpy(pretrained_wordemb))

    if args.use_cuda:
        # Keep large embedding on CPU when globals are on CPU
        text_encoder = text_encoder.to(globals_device)
        model = model.to(device)
        decoder = decoder.to(device)

    loss_fcn = nn.CrossEntropyLoss()

    if args.scale_grad_by_freq:
        embed_optimizer = optim.Adam(text_encoder.parameters(), lr=args.lr)
    else:
        embed_optimizer = optim.SparseAdam(list(text_encoder.parameters()), lr=args.lr)
    model_optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print(text_encoder)
    print(model)
    print(decoder)

    # --Training--
    def validate(text_encoder, model, decoder, g, subject_vs, targets):
        num_acc = 0.
        val_loss = 0.
        text_encoder.eval()
        model.eval()
        decoder.eval()
        for nf in NeighborSampler(g, args.test_batch_size,
                                  g.number_of_nodes(),
                                  neighbor_type='in',
                                  num_hops=n_layers,
                                  seed_nodes=subject_vs,
                                  add_self_loop=False,
                                  num_workers=args.workers):

            # Copy data from global graph
            node_embed_names = [['features']]
            for _ in range(n_layers):
                node_embed_names.append(['norm'])

            nf.copy_from_parent(node_embed_names=node_embed_names)

            with torch.no_grad():
                nf.apply_layer(0, lambda node: {'embed': text_encoder(node.data['features']).mean(1)})
                z = model(nf)
                pred = decoder(z)
                batch_nids = nf.layer_parent_nid(-1)
                batch_targets = targets[batch_nids]
                loss = loss_fcn(pred, batch_targets)
                num_acc += (torch.argmax(pred, dim=1) == batch_targets).sum().item()
                val_loss += loss.detach().item() * nf.layer_size(-1)

        accuracy = num_acc / len(subjects)
        val_loss = val_loss / len(subjects)

        return val_loss, accuracy

    # --real start of training--
    losses = []
    step = 0
    if args.early_stopping:
        # Init early stopping
        cnt_wait = 0
        best = 1e9
        best_t = 0

    for epoch in range(args.epochs):
        text_encoder.train()
        model.train()
        decoder.train()

        epoch_loss = 0.
        for nf in NeighborSampler(g, args.batch_size,
                                  num_neighbors,
                                  neighbor_type='in',
                                  shuffle=True,
                                  num_hops=n_layers,
                                  add_self_loop=False,
                                  seed_nodes=subject_vs,
                                  num_workers=args.workers):
            step += 1
            # Fill aggregate history from neighbors
            for i in range(n_layers):
                agg_history_str = 'agg_h_{}'.format(i)
                g.pull(nf.layer_parent_nid(i + 1), fn.copy_src(src='h_{}'.format(i), out='m'),
                       fn.sum(msg='m', out=agg_history_str),
                       lambda node: {agg_history_str: node.data[agg_history_str] * node.data['norm']}
                       )

            # Copy data from parent
            node_embed_names = [['features', 'h_0']]
            for i in range(1, n_layers):
                node_embed_names.append(['h_{}'.format(i), 'agg_h_{}.format(i-1)'])
            node_embed_names.append(['agg_h_{}'.format(n_layers - 1)])
            nf.copy_from_parent(node_embed_names=node_embed_names)

            # forward
            model_optimizer.zero_grad()
            embed_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            nf.apply_layer(0, lambda node: {'embed': text_encoder(node.data['features']).mean(1)})
            z = model(nf)  # z: test_batch_size x rep_size
            pred = decoder(z)  # test_batch_size x n_classes

            batch_nids = nf.layer_parent_nid(-1)  # indices in the all nodes (including patent and ipc)
            batch_targets = targets[batch_nids]
            loss = loss_fcn(pred, batch_targets)
            loss.backward()

            model_optimizer.step()
            embed_optimizer.step()
            decoder_optimizer.step()

            node_embed_names = [['h_{}'.format(i)] for i in range(n_layers)]
            node_embed_names.append([])

            nf.copy_from_parent(node_embed_names=node_embed_names)

            # Loss is sample-averaged
            epoch_loss += loss.detach().item() * nf.layer_size(-1)

        avg_epoch_loss = epoch_loss / len(subjects)

        # Now expand to all nodes for getting final represenations
        with open(os.path.join(args.out, MDS['loss_file']), 'a') as lossfile:
            print("{:.4f}".format(avg_epoch_loss), file=lossfile)

        if args.early_stopping:
            if avg_epoch_loss < best:
                best = avg_epoch_loss
                best_t = epoch
                cnt_wait = 0
                torch.save(text_encoder.state_dict(), os.path.join(args.out, 'best_text_encoder.pkl'))
                torch.save(model.state_dict(), os.path.join(args.out, 'best_model.pkl'))
                torch.save(decoder.state_dict(), os.path.join(args.out, 'best_decoder.pkl'))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print("Early stopping!")
                # Break out of main training loop if early-stopping criterion is met
                break

        if args.fastmode:
            # In fast mode, only evaluate accuracy in final epoch!
            print("Step {:7d} | Epoch {:4d} | Train Loss: {:.4f}".format(step,
                                                                         epoch,
                                                                         avg_epoch_loss))
            # Skip per-epoch computation of validation accuracy/loss
            continue

        val_loss, accuracy = validate(text_encoder, model, decoder, g, subject_vs, targets)
        with open(os.path.join(args.out, MDS['accuracy_file']), 'a') as accfile:
            print("Step {:7d} | Epoch {:4d} | Train loss: {:.4f} | Eval loss: {:.4f} | Accuracy {:.4f}"
                  .format(step, epoch, avg_epoch_loss, val_loss, accuracy), file=accfile)
        print("Step {:7d} | Epoch {:4d} | Train Loss: {:.4f} | Eval loss: {:.4f} | Accuracy {:.4f}"
              .format(step, epoch, avg_epoch_loss, val_loss, accuracy))

    if args.early_stopping:
        print('Loading {}th epoch'.format(best_t))
        text_encoder.load_state_dict(torch.load(os.path.join(args.out, 'best_text_encoder.pkl')))
        model.load_state_dict(torch.load(os.path.join(args.out, 'best_model.pkl')))
        decoder.load_state_dict(torch.load(os.path.join(args.out, 'best_decoder.pkl')))
        # For logging purposes
        epoch = best_t

    print("Shift models on cpu...")
    # PUT EVERYTHING STILL NEEDED ON CPU
    # WE DONT WANT TO RUN INTO MEM ISSUES HERE
    text_encoder = text_encoder.cpu()
    model = model.cpu()
    decoder = decoder.cpu()

    g.ndata['features'] = g.ndata['features'].cpu()
    g.ndata['norm'] = g.ndata['norm'].cpu()
    targets = targets.cpu()

    for i in range(n_layers):
        g.ndata.pop('h_{}'.format(i))
        g.ndata.pop('agg_h_{}'.format(i))

    # Put stuff in eval mode, no dropout and stuff
    text_encoder.eval()
    model.eval()
    decoder.eval()

    # and comp accuracy on the fly
    print("Computing final decoding accuracy")
    val_loss, accuracy = validate(text_encoder, model, decoder, g, subject_vs, targets)
    with open(os.path.join(args.out, MDS['accuracy_file']), 'a') as accfile:
        print("Step {:7d} | Epoch {:4d} | Eval loss: {:.4f} | Accuracy {:.4f}"
              .format(step, epoch, val_loss, accuracy), file=accfile)
    print("Step {:7d} | Epoch {:4d} | Eval loss: {:.4f} | Accuracy {:.4f}"
          .format(step, epoch, val_loss, accuracy))

    # Preprocess text encoding
    # Save representation\
    def embedding_fn(features, graph, node_ids):
        embedding = torch.zeros(graph.number_of_nodes(), rep_size)
        model.eval()
        graph.ndata['embed'] = features
        print("Computing norm...")
        print(graph)
        norm = 1. / graph.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0.
        graph.ndata['norm'] = norm

        for nf in NeighborSampler(graph, args.test_batch_size,
                                  graph.number_of_nodes(),  # all_patent_nodes
                                  neighbor_type='in',
                                  num_hops=n_layers,
                                  seed_nodes=node_ids,
                                  num_workers=args.workers,
                                  add_self_loop=True):

            node_embed_names = [['embed']]
            for _ in range(n_layers):
                node_embed_names.append(['norm'])
            nf.copy_from_parent(node_embed_names=node_embed_names)
            with torch.no_grad():
                z = model(nf)
                embedding[nf.layer_parent_nid(-1)] = z

        # Cleanup
        graph.ndata.pop('embed')
        graph.ndata.pop('norm')
        return embedding[node_ids]

    features = text_encoder(g.ndata.pop('features')).mean(1)
    print("g just before temporal embeddings (after dropping features):", g, sep='\n')

    concepts = data.ndata[data.ndata.type == SUBJECT_TYPE]["identifier"]
    concept_nids, descriptors = concepts.index.values, concepts.values

    print("Computing global embeddings...")
    # Global embedding
    representation = embedding_fn(features, g, concept_nids).numpy()
    print("Done.")
    return descriptors, representation, model, decoder, text_encoder, preprocessor, features
