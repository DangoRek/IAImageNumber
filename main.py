from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

digits = load_digits()

#alguns codigos só pra entender a biblioteca primeiro
#preenche o fundo da imagem, depois desenha os pixels com numero 1 e mostra a imagem
#plt.gray()
#plt.matshow(digits.images[1])
#plt.show()
#mostra o numero que representa a imagem
#print(digits.target[8])
#mostra a imagem em um array
#print(digits.images[8])

#1 camada de 500 neuronios com 1000 iterações (tava passando do limite 200), tol mais baixo pra conseguir ter mais iteracoes
#verbose ligado só pra ver o progresso e fazer os testes
net = MLPClassifier(verbose=True, hidden_layer_sizes=(500), max_iter=1000, tol=0.000000001)

#resultado de alguns testes que fiz, por algum motivo bizarro mais camadas e neuronios as vezes atrapalham o resultado
#0.00003939 1000 mas demora mt pra rodar
#0.00003956 500
#0.00004501 100
#0.00004724 50 50
#0.00005374 50 50 50
#0.00005841 100 100
#0.00006284 50 50 50 50
#0.00007529 100 100 100
#0.00007622 50

#separa os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=0)

#compara a imagem com o item do array resposta para treinamento
#net.fit(digits.data, digits.target)
net.fit(X_train, y_train)

#teste de precisao
#y_pred = net.predict(digits.data)
#print(accuracy_score(digits.target, y_pred))
y_pred = net.predict(X_test) #faz a previsao em um array de teste
#print(y_test, y_pred) #mostra o que o algoritmo acertou e errou
#compara o teste e a previsao pra descobrir a precisao
print(accuracy_score(y_test, y_pred))

#mostra o resultado do treinamento, só colocar o numero da imagem no [] e ele mostra o resultado
print(net.predict([digits.data[8]]))
