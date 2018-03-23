import tensorflow as tf

# tf.placeholder : 계산을 진행할 때 사용될 변수
# None은 크기가 지정되지 않았음을 의미함
X = tf.placeholder(tf.float32, [None, 3]) # 2차원 크기의 텐서를 만들고, 크기는 n, 2(이때 n은 변수)
print(X)

# X 플레이스 홀더에 넣을 값을 지정
# 설정했듯이 리스트의 두 번째 값이 3이었기 때문에 두번째 요소의 갯수는 3개이다.
X_data = [[1,2,3], [4,5,6]]

# tf.Variable : 그래프를 계산하며 최적화할 대상이 되는 변. 신경망에서 가장 중요한 변수다
# tf.random_normal : 정규 분포 그래프의 형태를 기반으로 랜덤한 값을 대입함
W = tf.Variable(tf.random_normal([3, 2])) # 3,2의 크기를 가지는 행렬 텐서를 만듬 - 값은 정규 분포에 기반한 랜덤 값
b = tf.Variable(tf.random_normal([2, 1])) # 2,1의 크기를 가지는 행렬 텐서를 만듬

# 값을 계산할 수식을 작성
# mat으로 시작하는 이름을 가진 함수는 행렬 계산을 수행하는 함수이다
expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("=== x_data ===")
print(X_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
# 수식을 실행하기 위해서는 값이 필요한데
# 이때 값은 feed_dict라는 argument를 이용해서 넣을 수 있다.
print(sess.run(expr, feed_dict={X: X_data})) # x_data라는 행렬과 W 행렬을 곱하고 b 행렬을 더한 결과값

sess.close()