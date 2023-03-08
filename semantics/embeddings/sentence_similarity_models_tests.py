{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "ef75eefe",
   "metadata": {},
   "source": [
    "# Models comparison for sentence similarity"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "034200c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sentence_transformers import SentenceTransformer, util\n",
    "\n",
    "from itertools import product, combinations"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "76cf777b",
   "metadata": {},
   "source": [
    "## ES"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "6fcb2d36",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "['¿Y que hombro y qué arte,',\n",
       " 'podrían retorcer la nervadura de tu corazón',\n",
       " 'Y cuando tu corazón comenzó a latir',\n",
       " '¿Qué formidable mano, qué formidables pies?',\n",
       " '¿Y qué hombro, y qué arte',\n",
       " 'pudo tejer la nervadura de tu corazón?',\n",
       " 'Y al comenzar los latidos de tu corazón,',\n",
       " '¿qué mano terrible? ¿Qué terribles pies?']"
      ]
     },
     "execution_count": 44,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "data = \"\"\"\n",
    "¿Y que hombro y qué arte,\n",
    "podrían retorcer la nervadura de tu corazón\n",
    "Y cuando tu corazón comenzó a latir\n",
    "¿Qué formidable mano, qué formidables pies?\n",
    "\n",
    "¿Y qué hombro, y qué arte\n",
    "pudo tejer la nervadura de tu corazón?\n",
    "Y al comenzar los latidos de tu corazón,\n",
    "¿qué mano terrible? ¿Qué terribles pies?\n",
    "\"\"\"\n",
    "texts = data.split('\\n')\n",
    "texts = [_ for _ in texts if _]\n",
    "texts"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "7d80af66",
   "metadata": {},
   "outputs": [],
   "source": [
    "comb = list(combinations(texts, 2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 129,
   "id": "3983852f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# https://huggingface.co/models?pipeline_tag=sentence-similarity&language=es&sort=likes\n",
    "test_models = [\n",
    "    \"symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli\",\n",
    "    \"clips/mfaq\",\n",
    "    \"hiiamsid/sentence_similarity_spanish_es\",\n",
    "    \"hackathon-pln-es/bertin-roberta-base-finetuning-esnli\",\n",
    "]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 130,
   "id": "1ed2976c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def comparison_matrix(embeddings):\n",
    "    length = len(embeddings)\n",
    "    matrix = np.zeros((length, length))\n",
    "\n",
    "    for i in range(length):\n",
    "        for j in range(length):\n",
    "            if i < j:  # ignore diagonal and lower-left redundant side\n",
    "                pair = embeddings[i], embeddings[j]\n",
    "                metric = util.cos_sim(*pair).numpy().squeeze()\n",
    "                metric = metric.item(0)  # scalar\n",
    "                matrix[i, j] = metric\n",
    "                # print(metric, pair)\n",
    "\n",
    "    return matrix"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 106,
   "id": "0106f00e",
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "8"
      ]
     },
     "execution_count": 106,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "embeddings = [sentence_model.encode(_) for _ in texts]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 125,
   "id": "b89103a5",
   "metadata": {},
   "outputs": [],
   "source": [
    "threshold = 0.5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 135,
   "id": "bd415b95",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "MODEL:  symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli\n",
      "[[0 0 0 0 1 0 0 0]\n",
      " [0 0 0 0 0 1 1 0]\n",
      " [0 0 0 0 0 0 1 0]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 1 0]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 0 0]]\n",
      "\t 0 4 ¿Y que hombro y qué arte,... \t ¿Y qué hombro, y qué arte...\n",
      "\t 1 5 podrían retorcer la nerva... \t pudo tejer la nervadura d...\n",
      "\t 1 6 podrían retorcer la nerva... \t Y al comenzar los latidos...\n",
      "\t 2 6 Y cuando tu corazón comen... \t Y al comenzar los latidos...\n",
      "\t 5 6 pudo tejer la nervadura d... \t Y al comenzar los latidos...\n",
      "\n",
      "MODEL:  clips/mfaq\n",
      "[[0 0 0 1 1 0 0 1]\n",
      " [0 0 1 1 0 1 1 1]\n",
      " [0 0 0 1 0 1 1 1]\n",
      " [0 0 0 0 1 1 1 1]\n",
      " [0 0 0 0 0 0 0 1]\n",
      " [0 0 0 0 0 0 1 1]\n",
      " [0 0 0 0 0 0 0 1]\n",
      " [0 0 0 0 0 0 0 0]]\n",
      "\t 0 3 ¿Y que hombro y qué arte,... \t ¿Qué formidable mano, qué...\n",
      "\t 0 4 ¿Y que hombro y qué arte,... \t ¿Y qué hombro, y qué arte...\n",
      "\t 0 7 ¿Y que hombro y qué arte,... \t ¿qué mano terrible? ¿Qué ...\n",
      "\t 1 2 podrían retorcer la nerva... \t Y cuando tu corazón comen...\n",
      "\t 1 3 podrían retorcer la nerva... \t ¿Qué formidable mano, qué...\n",
      "\t 1 5 podrían retorcer la nerva... \t pudo tejer la nervadura d...\n",
      "\t 1 6 podrían retorcer la nerva... \t Y al comenzar los latidos...\n",
      "\t 1 7 podrían retorcer la nerva... \t ¿qué mano terrible? ¿Qué ...\n",
      "\t 2 3 Y cuando tu corazón comen... \t ¿Qué formidable mano, qué...\n",
      "\t 2 5 Y cuando tu corazón comen... \t pudo tejer la nervadura d...\n",
      "\t 2 6 Y cuando tu corazón comen... \t Y al comenzar los latidos...\n",
      "\t 2 7 Y cuando tu corazón comen... \t ¿qué mano terrible? ¿Qué ...\n",
      "\t 3 4 ¿Qué formidable mano, qué... \t ¿Y qué hombro, y qué arte...\n",
      "\t 3 5 ¿Qué formidable mano, qué... \t pudo tejer la nervadura d...\n",
      "\t 3 6 ¿Qué formidable mano, qué... \t Y al comenzar los latidos...\n",
      "\t 3 7 ¿Qué formidable mano, qué... \t ¿qué mano terrible? ¿Qué ...\n",
      "\t 4 7 ¿Y qué hombro, y qué arte... \t ¿qué mano terrible? ¿Qué ...\n",
      "\t 5 6 pudo tejer la nervadura d... \t Y al comenzar los latidos...\n",
      "\t 5 7 pudo tejer la nervadura d... \t ¿qué mano terrible? ¿Qué ...\n",
      "\t 6 7 Y al comenzar los latidos... \t ¿qué mano terrible? ¿Qué ...\n",
      "\n",
      "MODEL:  hiiamsid/sentence_similarity_spanish_es\n",
      "[[0 0 0 0 1 0 0 0]\n",
      " [0 0 0 0 0 1 0 0]\n",
      " [0 0 0 0 0 0 1 0]\n",
      " [0 0 0 0 0 0 0 1]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 0 0]]\n",
      "\t 0 4 ¿Y que hombro y qué arte,... \t ¿Y qué hombro, y qué arte...\n",
      "\t 1 5 podrían retorcer la nerva... \t pudo tejer la nervadura d...\n",
      "\t 2 6 Y cuando tu corazón comen... \t Y al comenzar los latidos...\n",
      "\t 3 7 ¿Qué formidable mano, qué... \t ¿qué mano terrible? ¿Qué ...\n",
      "\n",
      "MODEL:  hackathon-pln-es/bertin-roberta-base-finetuning-esnli\n",
      "[[0 0 0 1 1 0 0 0]\n",
      " [0 0 1 0 0 1 1 0]\n",
      " [0 0 0 0 0 0 1 0]\n",
      " [0 0 0 0 1 0 0 1]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 1 0]\n",
      " [0 0 0 0 0 0 0 0]\n",
      " [0 0 0 0 0 0 0 0]]\n",
      "\t 0 3 ¿Y que hombro y qué arte,... \t ¿Qué formidable mano, qué...\n",
      "\t 0 4 ¿Y que hombro y qué arte,... \t ¿Y qué hombro, y qué arte...\n",
      "\t 1 2 podrían retorcer la nerva... \t Y cuando tu corazón comen...\n",
      "\t 1 5 podrían retorcer la nerva... \t pudo tejer la nervadura d...\n",
      "\t 1 6 podrían retorcer la nerva... \t Y al comenzar los latidos...\n",
      "\t 2 6 Y cuando tu corazón comen... \t Y al comenzar los latidos...\n",
      "\t 3 4 ¿Qué formidable mano, qué... \t ¿Y qué hombro, y qué arte...\n",
      "\t 3 7 ¿Qué formidable mano, qué... \t ¿qué mano terrible? ¿Qué ...\n",
      "\t 5 6 pudo tejer la nervadura d... \t Y al comenzar los latidos...\n"
     ]
    }
   ],
   "source": [
    "for model in test_models:\n",
    "    print('\\nMODEL: ', model)    \n",
    "    sentence_model = SentenceTransformer(model)\n",
    "    embeddings = [sentence_model.encode(_) for _ in texts]\n",
    "\n",
    "    matrix = comparison_matrix(embeddings)\n",
    "    boolean_matrix = (matrix >= threshold).astype(int)\n",
    "    # zero all values below threshold\n",
    "    matrix[matrix < threshold] = 0\n",
    "    print(boolean_matrix)\n",
    "    \n",
    "    for i in range(matrix.shape[0]):\n",
    "        for j in range(matrix.shape[1]):\n",
    "            if matrix[i, j] > threshold:\n",
    "                print('\\t', i, j, texts[i][:25] + '...', '\\t', texts[j][:25] + '...')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fb5dc77c",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
