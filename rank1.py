import sys
import re
import math

#inparams
#  featureFile: input file containing queries and url features
#return value
#  queries: map containing list of results for each query
#  features: map containing features for each (query, url, <feature>) pair
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

#inparams
#  queries: map containing list of results for each query
#  features: map containing features for each query,url pair
#return value
#  rankedQueries: map containing ranked results for each query
def baseline(queries, features, dfDict, totalDocNum):
    rankedQueries = {}

    # Parameters for BM25F counts for doc
    c_url = 4
    c_title = 3
    c_header = 2
    c_body = 0.2
    c_anchor = 6

    for query in queries:
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
      
      cos_scores = {}
      # doc tf (idf ignored for doc)
      urls = features[query]
      for url in urls:
        info = features[query][url]
        doc_vector = []
        body_length = info["body_length"] + 100
        for term in terms:
          tf_url = 0
          for i in range(0, len(url)-len(term)+1):
            if url[i:i+len(term)] == term:
              tf_url = tf_url + 1
          tf_title = 0
          if "title" in info:
            for word in info["title"].split(" "):
              if word == term:
                tf_title = tf_title + 1
          tf_header = 0
	  if "header" in info:
            for header in info["header"]:
              for word in header.split(" "):
                if word == term:
                  tf_header = tf_header + 1
          tf_body = 0
          if "body_hits" in info:
            if term in info["body_hits"]:
              tf_body = len(info["body_hits"][term])
          tf_anchor = 0
          if "anchors" in info:
            for text in info["anchors"]:
              count_per_anchor = 0
              for word in text.split(" "):
                if word == term:
                  count_per_anchor = count_per_anchor + 1
              tf_anchor = tf_anchor + count_per_anchor * info["anchors"][text]
          
          tf_total = c_url*tf_url + c_title*tf_title + c_header*tf_header + c_body*tf_body + c_anchor*tf_anchor
          tf_normal = (1.0 * tf_total) / body_length
          #tf_log = 0
          #if tf_normal > 0:
            #tf_log = 1 + math.log(tf_normal)
          doc_vector.append(tf_normal)

        cos_score = 0
        for i in range(0, len(terms)):
          cos_score = cos_score + doc_vector[i]*query_vector[i]
        cos_scores[url] = cos_score
      
      # Sort query results with cos_scores in decreasing order 
      rankedQueries[query] = sorted(results, key=lambda x: cos_scores[x], reverse=True)
    return rankedQueries

# getIdf gets returns a total number of doc and doc_freq_dict
def getIdf():
  term_id_f = "word.dict"
  posting_f = "posting.dict"
  doc_f = "doc.dict"

  allqueryFile = "AllQueryTerms"
  queryTermsDict = {}
  docNum = 0
  word_dict = {}
  doc_freq_dict = {}
  
  file = open(allqueryFile, 'r')
  for line in file.readlines():
    queryTermsDict[line.strip()] = 0

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

#inparams
#  queries: contains ranked list of results for each query
#  outputFile: output file name
def printRankedResults(queries):
    for query in queries:
      print("query: " + query)
      for res in queries[query]:
        print("  url: " + res)

#inparams
#  featureFile: file containing query and url features
def main(featureFile):
    #output file name
    outputFile = "ranked.txt" #Please don't change this!

    #populate map with features from file
    (queries, features) = extractFeatures(featureFile)

    #get idf values
    (totalDocNum, dfDict) = getIdf()

    #calling baseline ranking system, replace with yours
    rankedQueries = baseline(queries, features, dfDict, totalDocNum)
    
    #print ranked results to file
    printRankedResults(rankedQueries)
       
if __name__=='__main__':
    if (len(sys.argv) < 1):
      print "Insufficient number of arguments" 
    main(sys.argv[1])
