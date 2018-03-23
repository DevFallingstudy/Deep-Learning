# 텐서플로우의 기본적인 구성을 익히기 위한 example

import tensorflow as tf

# tf.constant 상수를 생성하는 함수입니다
hello = tf.constant('Hello, Tensorflow!')
print("python constant", hello)

# tf.add는 두 가지의 텐서를 더하는데에 사용된다. 일반 a + b 또한 사용 가능
a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
print(c)

# 해당 텐서를 실행 시키는 코드.
# 아직까지 변수를 실행 시킨다는 것에 대한 개념은 불확실하다
sess = tf.Session()
print(sess.run(hello))
print(sess.run([a, b, c]))

# 세션을 닫는다.
sess.close()