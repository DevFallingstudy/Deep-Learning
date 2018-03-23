# 털과 날개가 있는지 없는지에 따라 포유류인지 조류인지 분류하는 코드
# 털과 날개의 유무는 행렬 및 0과 1의 데이터로 표시
import tensorflow as tf
import numpy as np

# [털의 유무, 날개의 유무]
x_data = np.array([[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [기타, 포유류, 조류]
# 아래와 같이 하나의 값이 지정되고 지정된 값에 따라 특징이 변하는 데이터를 one-hot 데이터라고 한다.
y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 신경망 모델 구성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 신경망의 뉴런은 2차원 어레이로 작성된다.
# [입력층, 출력층]으로 나타내지며 아래와 같은 코드에선 사이즈 2의 입력층과 사이즈 3의 출력층을 가진다.
# 각 입출력 층의 예시는 위에서 선언한 x_data, y_data를 참고하면 된다.
# 털과 날개가 각각 결과값에 얼마만큼의 영향을 미치는지 모르기때문에 초기엔 랜덤 값을 정해줘야한다.
W = tf.Variable(tf.random_uniform([2, 3], -1., 1.))

# 편향의 사이즈는 해당하는 레이어의 출력층의 사이즈로 설정한다.
# 따라서 아래에선 3으로 설정(기타, 포유류, 조류)
b = tf.Variable(tf.zeros([3]))

# 신경망에 가중치 W와 편향 b를 적용
L = tf.add(tf.matmul(X, W), b)  # X * W + b
# 결과값에 활성화 함수인 ReLU 함수를 적용
L = tf.nn.relu(L)

# 마지막으로 softmax 함수를 이용하여 예측 결과값을 가공
# 일반적으로 출력값은 전체합이 들쑥날쑥하지만, softmax를 사용하면 합을 1로 만들어주기때문에
# 조금 더 직관적인 비교가 가능함
model = tf.nn.softmax(L)

# 모든 결과값에 대해서 평균값을 구한 뒤, 각 값을 더함으로써 계산을 최적화하고 진행할 수 있음
loss = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)

# 신경망 모델 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(900):
    sess.run(train_op, feed_dict={X: x_data, Y:y_data})
    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(loss, feed_dict={X: x_data, Y: y_data}))

# 결과 확인
# 0: 기타 1: 포유류 2: 조류
# tf.argmax : 예측 값과 정답 값의 행렬에서 가장 큰 값의 인덱스를 각각 가져옴
# 예) [[0 1 0] [1 0 0]] -> [1 0]
#    [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [1 0]
prediction = tf.argmax(model, 1) # 예측 결과 값의 행렬에서 가장 큰 값의 인덱스
target = tf.argmax(Y, 1) # 정답값의 행렬에서 가장 큰값의 인덱스
print("예측값 : ", sess.run(prediction, feed_dict={X:x_data}))
print("실제값 : ", sess.run(target, feed_dict={Y:y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도 : %.2f' % sess.run(accuracy * 100, feed_dict={X:x_data, Y:y_data}))