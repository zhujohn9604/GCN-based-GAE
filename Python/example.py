#%%
from train import *
from gensim.models import Word2Vec


#%% train the model
args = namespace(max_features=50000, max_length=256,
                 graphdir='./', use_authors=False,
                 collate_coauthorship=False, representation_size=256, lr=1e-3, weight_decay=0,
                 out='./', epochs=200, batch_size=128, test_batch_size=128,
                 embedding_dim=256, embedding_dropout=0, max_norm=None, num_neighbors=5,
                 n_layers=1, n_hidden=128, dropout=0.5, representation_dropout=0,
                 representation_activation=None, representation_layer_norm=False,
                 decoder_bias=False, fastmode=False, scale_grad_by_freq=False,
                 globals_on_cpu=True, early_stopping=False, patience=20,
                 warm_start=True, use_cuda=torch.cuda.is_available(),
                 load_pretrained_embedding=True, workers=4)


target_start_year = 2008
target_end_year = 2010
data = load_data(args.graphdir, target_start_year=target_start_year, target_end_year=target_end_year, supervised=False,
                 with_authors=args.use_authors,
                 collate_coauthorship=args.collate_coauthorship,
                 undirected=True)


text_features, preprocessor = preprocess_text(args, data)


descriptors, representation, model, decoder, text_encoder, preprocessor, features = embed_control_variate(args, data)
#%% save the results

model_PATH = 'model' + str(target_end_year) + 'all.pt'
torch.save(model, model_PATH)
text_encoder_PATH = 'text_encoder' + str(target_end_year) + 'all.pt'
torch.save(text_encoder, text_encoder_PATH)

np.save(r'representation_' + str(target_end_year) + 'all.npy', representation)
np.save(r'descriptor_' + str(target_end_year) + 'all.npy', descriptors)

#%% reload the results and calculate the transformed document vectors

target_start_year = 2008
target_end_year = 2010

data = load_data(args.graphdir, target_start_year=target_start_year, target_end_year=target_end_year, supervised=False,
                 with_authors=args.use_authors,
                 collate_coauthorship=args.collate_coauthorship,
                 undirected=True)


text_features, preprocessor = preprocess_text(args, data)



model_PATH = 'model' + str(target_end_year) + 'all.pt'
text_encoder_PATH = 'text_encoder' + str(target_end_year) + 'all.pt'

model = torch.load(model_PATH)
text_encoder = torch.load(text_encoder_PATH)
descriptors = np.load(r'descriptor_' + str(target_end_year) + 'all.npy', allow_pickle=True)
descriptors = np.array([i.rstrip() for i in descriptors])
representation = np.load(r'representation_' + str(target_end_year) + 'all.npy', allow_pickle=True)


import matplotlib.pyplot as plt
import pandas as pd


def transformed_vector(x):
    return (torch.matmul(W, x) + b).numpy()
def cs(a, b):
    return (a.dot(b)) / (np.linalg.norm(a) * np.linalg.norm(b))




W, b = list(model.layers[0].parameters())
W, b = W.detach(), b.detach()
doc_vectors = text_encoder(text_features).mean(1).detach()
transformed_doc = [transformed_vector(i) for i in doc_vectors]
#%%

rep_dict = dict(zip(descriptors, representation))
AI_term = ['machine learning', 'artificial intelligence', 'deep learning',
           'neural network', 'data mining', 'intelligence system', 'NLP']
DLT_term = ['distributed ledger', 'blockchain',
            'smart contract', 'bitcoin', 'cryptocurrency', 'encryption', 'ethereum']


def technological_ccordinates(doc_vector):
    AI_axis = np.max([cs(rep_dict[i], doc_vector)
                      for i in AI_term if i in rep_dict])
    DLT_axis = np.max([cs(rep_dict[i], doc_vector)
                       for i in DLT_term if i in rep_dict])
    return (AI_axis, DLT_axis)

tech_ccordinates = [technological_ccordinates(doc_vector) for doc_vector in transformed_doc]

AI_axes = [i[0] for i in tech_ccordinates]
DLT_axes = [i[1] for i in tech_ccordinates]


#%% plot
plt.figure(figsize=(12, 12))
plt.scatter(DLT_axes, AI_axes, s=1)
plt.xlabel('DLT intensity', fontsize=20)
plt.ylabel('AI intensity', fontsize=20)
plt.title('Year: ' + str(target_start_year) + '-' + str(target_end_year), fontsize=20)
plt.xlim(-0.2, 1)
#plt.xlim(0, 1)
plt.ylim(0, 1)
plt.tick_params(labelsize=20)
