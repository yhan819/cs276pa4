#!/usr/bin/env python

### Module imports ###
import sys
import math
import re
import numpy as np
from sklearn import linear_model, svm, preprocessing

def extractFeatures(featureFile):
    f = open(featureFile, 'r')
    queries = {}
    features = {}

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
    f.close()
    return (queries, features) 

def extractScores(rel_file):
  scores = []
  f = open(rel_file, 'r')
  for line in f:
    if "url" in line:
      s = line.split()
      scores.append(float(s[-1]))
  return scores


# getIdf gets returns a total number of doc and doc_freq_dict
def getIdf():
  term_id_f = "word.dict"
  posting_f = "posting.dict"
  doc_f = "doc.dict"

  #allqueryFile = "AllQueryTerms"
  queryTermsDict = {}
  docNum = 0
  word_dict = {}
  doc_freq_dict = {}

  file = open(doc_f, 'r')
  for l in file.readlines():
    docNum += 1
  
  file = open(term_id_f, 'r')
  for line in file.readlines():
    parts = line.split('\t')
    word_dict[int(parts[1])] = parts[0]
  file = open(posting_f, 'r') 
  for line in file.readlines():
    parts = line.split('\t')
    term_id = int(parts[0])
    doc_freq = int(parts[2])
    doc_freq_dict[word_dict[term_id]] = doc_freq
  
  return (docNum, doc_freq_dict)

def get_feature_vecs(queries, features, dfDict, totalDocNum, task):
    result = []
    index = 0
    index_map = {}

    for query in queries:
      index_map[query] = {}
      results = queries[query]
      # query idf (tf not needed. All 1's)
      terms = query.split(" ")
      query_idf_list = []
      for term in terms:
        df = 1
        if term in dfDict:
          df = df + dfDict[term]
        idf = math.log((1.0 *(totalDocNum + 1)) / df)#math.log((totalDocNum + 1)/df)
        query_idf_list.append(idf)
      query_vector = query_idf_list
      
      # doc tf (idf ignored for doc)
      urls = features[query]
      for url in urls:
        index_map[query][url] = index
        index += 1
        info = features[query][url]
        doc_vector = []

        url_vec = []
        title_vec = []
        header_vec = []
        body_vec = []
        anchor_vec = []
        body_length = info["body_length"] + 100
        pagerank = info['pagerank']
        pdf = 0
        if url[len(url)-4:len(url)] == ".pdf":
          pdf = 1

        for term in terms:
          #url
          tf_url = 0
          for i in range(0, len(url)-len(term)+1):
            if url[i:i+len(term)] == term:
              tf_url = tf_url + 1
          if tf_url == 0 and task == 1:
            tf_url = 1
          #tf_title
          tf_title = 0
          if "title" in info:
            for word in info["title"].split(" "):
              if word == term:
                tf_title = tf_title + 1
          if tf_title == 0 and task == 1:
            tf_title = 1
          #tf_header
          tf_header = 0
	  if "header" in info:
            for header in info["header"]:
              for word in header.split(" "):
                if word == term:
                  tf_header = tf_header + 1
          if tf_header == 0 and task == 1:
            tf_header = 1
          #tf_body
          tf_body = 0
          if "body_hits" in info:
            if term in info["body_hits"]:
              tf_body = len(info["body_hits"][term])
          if tf_body == 0 and task == 1:
            tf_body = 1
          #tf_anchor
          tf_anchor = 0
          if "anchors" in info:
            for text in info["anchors"]:
              count_per_anchor = 0
              for word in text.split(" "):
                if word == term:
                  count_per_anchor = count_per_anchor + 1
              tf_anchor = tf_anchor + count_per_anchor * info["anchors"][text]
          if tf_anchor == 0 and task == 1:
            tf_anchor = 1
          #Task 1: used scaling
          if task == 1:
            url_vec.append(1.0 * math.log(tf_url) )
            title_vec.append(1.0 * math.log(tf_title) )
            header_vec.append(1.0 * math.log(tf_header) )
            body_vec.append(1.0 * math.log(tf_body) )
            anchor_vec.append(1.0 * math.log(tf_anchor) )
          #Task 2: used normalization
          elif task == 2 or task == 3:
            url_vec.append(1.0 * (tf_url) / body_length)
            title_vec.append(1.0 * (tf_title) / body_length)
            header_vec.append(1.0 * (tf_header) / body_length)
            body_vec.append(1.0 * (tf_body) / body_length)
            anchor_vec.append(1.0 * (tf_anchor) / body_length)
          #tf_log = 0
          #if tf_normal > 0:
            #tf_log = 1 + math.log(tf_normal)
        total_vecs = [url_vec, title_vec, header_vec, body_vec, anchor_vec]
        for i in range(0, 5):
          tfidf = 0.0
          for j in range(0, len(terms)):
            tfidf += query_vector[j] * total_vecs[i][j]
          doc_vector.append(tfidf)
        num_pdf = 0
        if task == 3:
          doc_vector.append(pdf)
          print >> sys.stderr, pdf
          #doc_vector.append(pagerank)
        result.append(doc_vector)
      
    return result, index_map

def pair_docs(f_vecs, scores, queries, index_map):
  f_vecs = preprocessing.scale(f_vecs)
  pairs = []
  y = []
  for q in queries:
    urls = queries[q]
    for i in range(0, len(urls)):
      for j in range(i, len(urls)):
        a = f_vecs[index_map[q][urls[i]]]
        b = f_vecs[index_map[q][urls[j]]]
        tmp = []
        for k in range(0, len(a)):
          tmp.append(0.0 + a[k] - b[k])
        pairs.append(tmp)
        if scores[index_map[q][urls[i]]] > scores[index_map[q][urls[j]]]:
          score = 1
        elif scores[index_map[q][urls[i]]] < scores[index_map[q][urls[j]]]:
          score = -1
        else:
            score = -1
        #score = 1 if scores[index_map[q][urls[i]]] > scores[index_map[q][urls[j]]] else -1
        y.append(score)
  return (pairs, y)
          

###############################
##### Point-wise approach #####
###############################
def pointwise_train_features(train_data_file, train_rel_file):
  (queries, features) = extractFeatures(train_data_file)
  scores = extractScores(train_rel_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 1)
  return (f_vecs, scores)
 
def pointwise_test_features(test_data_file):
  (queries, features) = extractFeatures(test_data_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 1)
  
  # index_map[query][url] = i means X[i] is the feature vector of query and url

  return (f_vecs, queries, index_map)
 
def pointwise_learning(X, y):
  # stub, you need to implement
  model = linear_model.LinearRegression()
  model.fit(X,y)
  return model

def pointwise_testing(X, model):
  # stub, you need to implement
  y = model.predict(X)
  return y

##############################
##### Pair-wise approach #####
##############################
def pairwise_train_features(train_data_file, train_rel_file):
  (queries, features) = extractFeatures(train_data_file)
  scores = extractScores(train_rel_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 2)
  (X, y) = pair_docs(f_vecs, scores, queries, index_map)
  
  return (X, y)

def pairwise_test_features(test_data_file):
  (queries, features) = extractFeatures(test_data_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 2) 
  # stub, you need to implement
  # index_map[query][url] = i means X[i] is the feature vector of query and url
  # RIGHT NOW SCALING 
  #f_vecs = preprocessing.scale(f_vecs)

  return (f_vecs, queries, index_map)

def pairwise_learning(X, y):
  # stub, you need to implement
  model = svm.SVC(kernel='linear', C=1.0)
  model.fit(X,y)
  return model

def pairwise_testing(X, model):
  # stub, you need to implement
  coefs = model.coef_[0]
  y = []
  for x in X:
    score = 0.0
    for i in range(0, len(x)):
      score += 1.0 * x[i] * coefs[i]
    y.append(score)
  
  return y

#####################################################
##### Pairwise with additional features approach #####
#####################################################
def pairwise_train_features_add(train_data_file, train_rel_file):
  (queries, features) = extractFeatures(train_data_file)
  scores = extractScores(train_rel_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 3)
  (X, y) = pair_docs(f_vecs, scores, queries, index_map)
  
  return (X, y)

def pairwise_test_features_add(test_data_file):
  (queries, features) = extractFeatures(test_data_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 3)
  # stub, you need to implement
  # index_map[query][url] = i means X[i] is the feature vector of query and url
  # RIGHT NOW SCALING 
  #f_vecs = preprocessing.scale(f_vecs)

  return (f_vecs, queries, index_map)

####################
##### Training #####
####################
def train(train_data_file, train_rel_file, task):
  sys.stderr.write('\n## Training with feature_file = %s, rel_file = %s ... \n' % (train_data_file, train_rel_file))
  
  if task == 1:
    # Step (1): construct your feature and label arrays here
    (X, y) = pointwise_train_features(train_data_file, train_rel_file) 
    # Step (2): implement your learning algorithm here
    model = pointwise_learning(X, y)
  elif task == 2:
    # Step (1): construct your feature and label arrays here
    (X, y) = pairwise_train_features(train_data_file, train_rel_file)
    # Step (2): implement your learning algorithm here
    model = pairwise_learning(X, y)
  elif task == 3: 
    # Add more features
    (X, y) = pairwise_train_features_add(train_data_file, train_rel_file)
    model = pairwise_learning(X, y)
  elif task == 4: 
    # Extra credit 
    print >> sys.stderr, "Extra Credit\n"
  else: 
    X = [[0, 0], [1, 1], [2, 2]]
    y = [0, 1, 2]
    model = linear_model.LinearRegression()
    model.fit(X, y)
  
  # some debug output
  #weights = model.coef_
  #print >> sys.stderr, "Weights:", str(weights)

  return model 

###################
##### Testing #####
###################
def test(test_data_file, model, task):
  sys.stderr.write('\n## Testing with feature_file = %s ... \n' % (test_data_file))

  if task == 1:
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pointwise_test_features(test_data_file)
    
    # Step (2): implement your prediction code here
    y = pointwise_testing(X, model)
  elif task == 2:
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pairwise_test_features(test_data_file)
    
    # Step (2): implement your prediction code here
    y = pairwise_testing(X, model)
  elif task == 3: 
    # Add more features
    # Step (1): construct your test feature arrays here
    (X, queries, index_map) = pairwise_test_features_add(test_data_file)
    
    # Step (2): implement your prediction code here
    y = pairwise_testing(X, model)
 
  elif task == 4: 
    # Extra credit 
    print >> sys.stderr, "Extra credit\n"
  else:
    queries = ['query1', 'query2']
    index_map = {'query1' : {'url1':0}, 'query2': {'url2':1}}
    X = [[0.5, 0.5], [1.5, 1.5]]  
    y = model.predict(X)
  
  # some debug output
  for query in queries:
    result = []
    print >> sys.stdout, "query: %s" % query 
    for url in index_map[query]:
      result.append((url, y[index_map[query][url]]))
      #print >> sys.stderr, "Query:", query, ", url:", url, ", value:", y[index_map[query][url]]
    result = sorted(result, key=lambda x: x[1], reverse=True)
    for r in result:
      print >> sys.stdout, "  url: %s" % r[0]
    

  # Step (3): output your ranking result to stdout in the format that will be scored by the ndcg.py code

if __name__ == '__main__':
  sys.stderr.write('# Input arguments: %s\n' % str(sys.argv))
  
  if len(sys.argv) != 5:
    print >> sys.stderr, "Usage:", sys.argv[0], "train_data_file train_rel_file test_data_file task"
    sys.exit(1)
  
  train_data_file = sys.argv[1]
  train_rel_file = sys.argv[2]
  test_data_file = sys.argv[3]
  task = int(sys.argv[4])
  print >> sys.stderr, "### Running task", task, "..."
  
  model = train(train_data_file, train_rel_file, task)
  test(test_data_file, model, task)
