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

    count = {}
    count["num_url"] = 0
    count["len_url"] = 0
    count["len_title"] = 0
    count["len_header"] = 0
    count["len_body"] = 0
    count["len_anchor"] = 0

    for line in f:
      key = line.split(':', 1)[0].strip()
      value = line.split(':', 1)[-1].strip()
      if(key == 'query'):
        query = value
        queries[query] = []
        features[query] = {}
      elif(key == 'url'):
        url = value
        count["num_url"] += 1
        count["len_url"] += len(value)
        queries[query].append(url)
        features[query][url] = {}
      elif(key == 'title'):
        features[query][url][key] = value
        count["len_title"] += len(value.split())
      elif(key == 'header'):
        curHeader = features[query][url].setdefault(key, [])
        curHeader.append(value)
        features[query][url][key] = curHeader
        count["len_header"] += len(value.split())
      elif(key == 'body_hits'):
        if key not in features[query][url]:
          features[query][url][key] = {}
        temp = value.split(' ', 1)
        features[query][url][key][temp[0].strip()] \
                    = [int(i) for i in temp[1].strip().split()]
      elif(key == 'body_length' or key == 'pagerank'):
        features[query][url][key] = int(value)
        if key == 'body_length':
          count["len_body"] += int(value)
      elif(key == 'anchor_text'):
        anchor_text = value
        if 'anchors' not in features[query][url]:
          features[query][url]['anchors'] = {}
      elif(key == 'stanford_anchor_count'):
        features[query][url]['anchors'][anchor_text] = int(value)
        count["len_anchor"] += len(anchor_text.split()) * int(value)
    f.close()
    return (queries, features, count) 

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

def makePostingList(query, text, isURL):
  postingLists = []
  query_terms = query.split(" ")
  if isURL == False:
    text_terms = text.split(" ")
    for i in range(0, len(query_terms)):
      query_term = query_terms[i]
      posting = []
      for j in range(0, len(text_terms)):
        if text_terms[j] == query_term:
          posting.append(j)
      postingLists.append(posting)
    return (postingLists, -1)
  else:
    tot_length = 0
    for i in range(0, len(query_terms)):
      query_term = query_terms[i]
      tot_length = tot_length + len(query_term)
      posting = []
      for j in range(0, len(text) - len(query_term) + 1):
        if text[j:j+len(query_term)] == query_term:
          posting.append(j)
      postingLists.append(posting)
    return (postingLists, tot_length)

# Only for more than 2 word phrase queries
# And all terms in the query must be present (each term appears in the feature text)
def findWindow(numTerms, postingLists):
  #print >> sys.stderr, postingLists
  termIndexes = []
  allContained = True
  for i in range(0, numTerms):
    termIndexes.append(0)

  first_term_index = 0
  first_term = float("inf")
  if len(postingLists[0]) != 0:
    first_term = postingLists[0][first_term_index]
  (termIndexes, hasBigger) = shiftPostingIndexes(first_term, postingLists, termIndexes)
  if hasBigger == False:
    return float("inf")
  window_size = postingLists[len(postingLists)-1][termIndexes[len(termIndexes)-1]] - postingLists[0][termIndexes[0]]
  first_term_index = first_term_index + 1
  hasBigger = True
  while first_term_index < len(postingLists[0]):
    termIndexes[0] = first_term_index
    (termIndexes, hasBigger) = shiftPostingIndexes(postingLists[0][first_term_index], postingLists, termIndexes)
    if hasBigger == False:
      break
    first_term_index = first_term_index + 1
    new_window_size = postingLists[len(postingLists)-1][termIndexes[len(termIndexes)-1]] - postingLists[0][termIndexes[0]]
    if new_window_size < window_size:
      window_size = new_window_size
  return window_size+1


def shiftPostingIndexes(newSmaller, postingLists, termIndexes):
  hasBigger = True
  # From 2nd term posting list
  for i in range(1, len(postingLists)):
    index = termIndexes[i]
    while index < len(postingLists[i]):
      if postingLists[i][index] >= newSmaller:
        termIndexes[i] = index
        newSmaller = postingLists[i][index]
        break
      index = index + 1
    if index == len(postingLists[i]):
      hasBigger = False
      break
  return (termIndexes, hasBigger)

def get_feature_vecs(queries, features, dfDict, totalDocNum, task, count):
    result = []
    index = 0
    index_map = {}
    _pdf = True
    _pagerank = False
    _window = True
    _bmf = True
    _body = False

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
        if task == 1:
          idf = math.log((1.0 * (totalDocNum + 1)) / df)
        else:
          idf = (1.0 * (totalDocNum + 1))/df
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

        total_bodyhit = 0
        pagerank = info['pagerank']
        pdf = 0
        if url[len(url)-4:len(url)] == ".pdf":
          pdf = 1
        window_sizes = [float("inf"), float("inf"), float("inf"), float("inf"), float("inf")]
	
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
	      total_bodyhit += tf_body
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
            url_vec.append(1.0 * (tf_url))
            title_vec.append(1.0 * (tf_title))
            header_vec.append(1.0 * (tf_header))
            body_vec.append(1.0 * (tf_body))
            anchor_vec.append(1.0 * (tf_anchor)) 
        total_vecs = [url_vec, title_vec, header_vec, body_vec, anchor_vec]
        for i in range(0, 5):
          tfidf = 0.0
          for j in range(0, len(terms)):
            tfidf += query_vector[j] * total_vecs[i][j]
	  if i != 3:
            doc_vector.append(tfidf)
          else:
 	    if task != 3:
	      doc_vector.append(tfidf)
        if task == 3:
          if _pdf:
            doc_vector.append(pdf)
          if _pagerank:
            doc_vector.append(pagerank)
          
          # url
          #url_contained = allInText(query, url)
          #if len(url_contained) == len(terms):
          (url_postingLists, tot_length) = makePostingList(query, url, True)
          window_size = float("inf")
          if len(terms) > 1:
            window_size = findWindow(len(terms), url_postingLists)
          if window_size is not float("inf"):
            window_size = window_size + len(terms[len(terms)-1])-1 #for url, char length
          window_sizes[0] = window_size
          # title
          if "title" in info:
            title = info["title"]
            #title_contained = allInText(query, title)
            #if len(title_contained) == len(terms):
            (title_postingLists, flag) = makePostingList(query, title, False)
            window_size = float("inf")
            if len(terms) > 1:
              window_size = findWindow(len(terms), title_postingLists)
            window_sizes[1] = window_size
          # header
          if "header" in info:
            headers = info["header"]
            smallest_window_hd = window_sizes[2]
            header_allPostingLists = []
            for header in headers:
              #header_contained = allInText(query, header)
              #if len(header_contained) == len(terms):
              (header_postingLists, flag) = makePostingList(query, header, False)
              header_allPostingLists.append(header_postingLists)
              window_size = float("inf")
              if len(terms) > 1:
                window_size = findWindow(len(terms), header_postingLists)
              if window_size < smallest_window_hd:
                smallest_window_hd = window_size
            window_sizes[2] = smallest_window_hd
          # body
          if "body_hits" in info:
            body_postingLists = []
            for i in range(0, len(terms)):
              posting = []
              if terms[i] in info["body_hits"].keys():
                for x in range(0, len(info["body_hits"][terms[i]])):
                  posting.append(info["body_hits"][terms[i]][x])
              body_postingLists.append(posting)
            window_size = float("inf")
            if len(terms) > 1:
              window_size = findWindow(len(terms), body_postingLists)
            window_sizes[3] = window_size
          # anchor
          if "anchors" in info:
            smallest_window = window_sizes[4]
            anchor_allPostingLists = []
            for text in info["anchors"].keys():
              #anchor_contained = allInText(query, text)
              #if len(anchor_contained) == len(terms):
              (anchor_postingLists, flag) = makePostingList(query, text, False)
              anchor_allPostingLists.append(anchor_postingLists)
              window_size = float("inf")
              if len(terms) > 1:
                window_size = findWindow(len(terms), anchor_postingLists)
              if window_size < smallest_window:
                smallest_window = window_size
            window_sizes[4] = smallest_window
          
          minWindow = 0
          for i in range(1, len(window_sizes)):
            if i == 4 and window_sizes[i] == len(terms):
	      minWindow = 1
          if window_sizes[0] <= len(query) - len(terms) + 4:
 	    minWindow = 1.5
          #print >> sys.stderr, minWindow
          if _window:
	    doc_vector.append(minWindow)  
        
          avgurl = count["len_url"] * 1.0 / count["num_url"]
          avgtitle = count["len_title"] * 1.0 / count["num_url"]
          avgheader = count["len_header"] * 1.0 / count["num_url"]
          avgbody = count["len_body"] * 1.0 / count["num_url"]
          avganchor = count["len_anchor"] * 1.0 / count["num_url"]

          # Parameters for tf counts for doc
          b_url = 0.2
          b_title = 0.5
          b_header = 0.5
          b_body = 0.4
          b_anchor = 0.2
          w_url = 4
          w_title = 3
          w_header = 2
          w_body = 0.2
          w_anchor = 6
          lamb = 1
          lamb_p = 1
          k_1 = 3

          doc_score = 0.0
          info = features[query][url]
          for t in terms:
            wdt = 0.0
            #ftf for url
            fturl = 0.0
            tfurl = 0
            for i in range(0, len(url)-len(t)+1):
              if url[i:i+len(t)] == t:
                tfurl += 1
            fturl += 1.0 * tfurl / (1 + b_url * ((len(url) / avgurl) - 1))
            wdt += w_url * fturl
            #ftf for title
            fttitle = 0.0
            tftitle = 0
            t_l = info["title"].split()
            for word in t_l:
              if word == t:
                tftitle += 0
            if len(t_l) > 0:
              fttitle += 1.0 * tftitle / (1 + b_title * ((len(t_l) / avgtitle) - 1))
              wdt += w_title * fttitle
            #tft for header
            if "header" in info:
              ftheader = 0.0
              tfheader = 0
              lenhead = 0
              for header in info["header"]:
                lenhead += len(header.split())
                for word in header.split():
                  if word == t:
                    tfheader += 1
              ftheader += 1.0 * tfheader / (1 + b_header * ((lenhead / avgheader) - 1))
              wdt += w_header * ftheader
            # ftf for body
            if "body_hits" in info:
              ftbody = 0.0
              tfbody = 0
              if t in info["body_hits"]:
                tfbody = len(info["body_hits"][t])
              ftbody += 1.0 * tfbody / (1 + b_body * ((info["body_length"] / avgbody) - 1))
              wdt += w_body * ftbody
            # ftf for anchor
            if "anchors" in info:
              ftanchor = 0.0
              tfanchor = 0
              anchor_len = 0
              for text in info["anchors"]:
                c_p_a = 0
                for word in text.split():
                  if word == t:
                    c_p_a += 1
                tfanchor += c_p_a * info["anchors"][text]
                anchor_len += len(text.split()) * info["anchors"][text]
              ftanchor += 1.0 * tfanchor / (1 + b_anchor * ((anchor_len / avganchor) - 1))
              wdt += w_anchor * ftanchor
             
            #idf
            if t not in dfDict:
              df = 1
            else:
              df = dfDict[t] + 1
            idf = math.log10((totalDocNum + 1)/df)
            doc_score += wdt * idf / (k_1 + wdt)

          #nontextual: pagerank
          #nont = lamb * 1.0 * math.log(lamb_p + info["pagerank"])
          #nont = lamb * info["pagerank"] / (lamb_p + info["pagerank"])
          #nont = lamb / (lamb_p + math.exp(-1 * info["pagerank"] * lamb_p))
          nont = 0
          bmf_score = doc_score + nont
          if _bmf:
            doc_vector.append(bmf_score)
	  if _body:
	    doc_vector.append((1.0*total_bodyhit)/body_length)	  
 
        result.append(doc_vector)

    return result, index_map

def pair_docs(f_vecs, scores, queries, index_map):
  f_vecs = preprocessing.scale(f_vecs)
  pairs = []
  y = []
  num1 = 0
  numm1 = 0
  for q in queries:
    urls = queries[q]
    for i in range(0, len(urls)):
      for j in range(i, len(urls)):
        a = f_vecs[index_map[q][urls[i]]]
        b = f_vecs[index_map[q][urls[j]]]
        tmp = []
        for k in range(0, len(a)):
          tmp.append(0.0 + a[k] - b[k])
        #pairs.append(tmp)
        if scores[index_map[q][urls[i]]] > scores[index_map[q][urls[j]]]:
          score = 1
	  y.append(score)
	  pairs.append(tmp)
        elif scores[index_map[q][urls[i]]] < scores[index_map[q][urls[j]]]:
          score = -1
          y.append(score)
	  pairs.append(tmp)
        else:
          score = 1
  return (pairs, y)
          

###############################
##### Point-wise approach #####
###############################
def pointwise_train_features(train_data_file, train_rel_file):
  (queries, features, count) = extractFeatures(train_data_file)
  scores = extractScores(train_rel_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 1, count)
  return (f_vecs, scores)
 
def pointwise_test_features(test_data_file):
  (queries, features, count) = extractFeatures(test_data_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 1, count)
  
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
  (queries, features, count) = extractFeatures(train_data_file)
  scores = extractScores(train_rel_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 2, count)
  (X, y) = pair_docs(f_vecs, scores, queries, index_map)
  
  return (X, y)

def pairwise_test_features(test_data_file):
  (queries, features, count) = extractFeatures(test_data_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 2, count) 
  # stub, you need to implement
  # index_map[query][url] = i means X[i] is the feature vector of query and url
  # RIGHT NOW SCALING 
  f_vecs = preprocessing.scale(f_vecs)

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
  (queries, features, count) = extractFeatures(train_data_file)
  scores = extractScores(train_rel_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 3, count)
  (X, y) = pair_docs(f_vecs, scores, queries, index_map)
  
  return (X, y)

def pairwise_test_features_add(test_data_file):
  (queries, features, count) = extractFeatures(test_data_file)
  (docNum, doc_freq_dict) = getIdf()
  (f_vecs, index_map) = get_feature_vecs(queries, features, doc_freq_dict, docNum, 3, count)
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
  weights = model.coef_
  print >> sys.stderr, "Weights:", str(weights)

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
