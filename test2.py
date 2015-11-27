import exact_surface_matching
import token_template
import entity_labels
import json
import entity_base_scoring
import entity_linking
import output
import sys



sys.setrecursionlimit(5000)
text='이 사자성어는 ‘기구를 편리하게 쓰고 먹을 것과 입을 것을 넉넉하게 하여, 국민의 생활을 나아지게 함’을 의미한다. 조선후기 박지원, 박제가 등 북학파 실학자들이 주장한 이념을 표현한 말인데, 이 사자성어는 무엇일까?'

#print("morph analysis...")
#tokens = token_template.Tokens()
#tokens.Add_token(0,6,"덕혜",["NNP"],0,True,False)
#tokens.Add_token(6,12,"옹주",["NNG"],0,False,False)
#tokens.Add_token(12,15,"는",["JX"],0,False,True)
#tokens.Add_token(16,22,"조선",["NNP"],0,True,False)
#tokens.Add_token(22,25,"의",["JKG"],0,False,True)
#tokens.Add_token(26,29,"제",["XPN"],0,True,False)
#tokens.Add_token(29,31,"26",["SN"],0,False,False)
#tokens.Add_token(31,34,"대",["NNB"],0,False,True)
#tokens.Add_token(35,38,"왕",["NNG"],0,True,False)
#tokens.Add_token(38,41,"이",["VCP"],0,False,False)
#tokens.Add_token(41,44,"자",["EC"],0,False,True)
#tokens.Add_token(45,51,"대한",["NNG"],0,True,False)
#tokens.Add_token(51,57,"제국",["NNG"],0,False,False)
#tokens.Add_token(57,60,"의",["JKG"],0,False,True)
#tokens.Add_token(61,67,"초대",["NNG"],0,True,True)
#tokens.Add_token(68,74,"황제",["NNG"],0,True,True)
#tokens.Add_token(75,81,"고종",["NNG"],0,True,False)
#tokens.Add_token(81,84,"과",["JC"],0,False,True)
#tokens.Add_token(85,88,"귀",["NNG"],0,True,False)
#tokens.Add_token(88,91,"인",["VCP", "ETM"],0,False,True)
#tokens.Add_token(92,95,"양",["NNP"],0,True,False)
#tokens.Add_token(95,98,"씨",["NNB"],0,False,False)
#tokens.Add_token(98,101,"의",["JKG"],0,False,True)
#tokens.Add_token(102,108,"황녀",["NNG"],0,True,False)
#tokens.Add_token(108,111,"이",["VCP"],0,False,False)
#tokens.Add_token(111,114,"다",["EF"],0,False,False)
#tokens.Add_token(114,115,".",["SF"],0,False,True)

with open('ko-config.json') as data_file:    
    config = json.load(data_file)
entity_linking.Initialize(config)

print("morph analysis...")
tokens = entity_linking.Config.morph_analysis.Parse_text(text)

print("finding entity candidates...")
matches = exact_surface_matching.Find_entity_candidates(tokens, overlapping=True)

print("retrieving uri base scores...")
entity_base_scores = entity_base_scoring.Entity_base_scores(matches)

print("initiating entity linking...")
entity_detection = entity_linking.Entity_detection(tokens, matches, entity_base_scores, lower_bound=0.5)

result = output.Output(entity_detection)
#print(result.Response_repr())
print("done...")
