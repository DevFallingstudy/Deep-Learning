import tensorflow as tf
import numpy as np

x_data = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 1],
    ]
)

y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 1],
])

# 신경망 모델 구성
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# [입력 값의 사이즈, 히든 레이어의 뉴런 개수] - >2, 10
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
# [히든 레이어의 뉴런 개수, 결과 값의 사이즈] - >2, 10
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

# 각 레이어의 출력 값의 사이즈
b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# L1 = X * W1 + b1(with relu)
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# L2 = model = L1 * W2 + b2
model = tf.add(tf.matmul(L1, W2), b2)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(loss)


# 학습
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X:x_data, Y:y_data})

    print(step+1, sess.run(loss, feed_dict={X:x_data, Y:y_data}))

#결과 확인
prediction = tf.argmax(model, 1)
target = tf.argmax(y_data, 1)
print("예측값 :", sess.run(prediction, feed_dict={X: x_data}))
print("실제값 :", sess.run(target, feed_dict={Y: y_data}))

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("정확도 : %.2f" % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
