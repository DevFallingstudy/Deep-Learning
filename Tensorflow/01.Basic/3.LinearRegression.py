import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

# name : 나중에 텐서의 값을 추적하거나 살펴보기위해 이름을 붙여주는 코드
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
print(X)
print(Y)

# X와 Y의 상관 관계를 분석하기 위한 가설 수식 작성
# y(hypothesis) = W * x + b
# hypothesis = tf.add(tf.multiply(W, X), b)
hypothesis = W * X + b


# 손실 함수(loss function) 작성
# loss을 확인 - 각 X 축에 대한 Y(=hypothesis) 값을 정답 Y(=Y)값과 비교
loss = tf.reduce_mean(tf.square(hypothesis - Y))
# 경사 하강법을 통해 최적화를 수행할 수 있도록 지정
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 경사 하강법을 사용하는 optimizer를 사용하여 loss를 최소화하는 방안을 찾도록 train 텐서를 생성
train_op = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 최적화를 100번 수행
    for step in range(500):
        _, cost_val = sess.run([train_op, loss], feed_dict={X: x_data, Y: y_data})

        print(step, cost_val, sess.run(W), sess.run(b))

    # 최적화 테스트
    print("\n=== Test ===")
    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))