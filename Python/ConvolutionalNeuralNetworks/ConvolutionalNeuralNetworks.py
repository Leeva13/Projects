import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imageio

# Функції для обробки зображень
def load_and_process_img(path):
    img = Image.open(path).resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return tf.convert_to_tensor(img, dtype=tf.float32)

def deprocess_img(img):
    img = img.copy()
    img[:, :, 0] += 103.939  # Відновлення каналу B
    img[:, :, 1] += 116.779  # Відновлення каналу G
    img[:, :, 2] += 123.68   # Відновлення каналу R
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Завантаження зображень
content_img = load_and_process_img('cat.jpg')
style_img = load_and_process_img('vangogh.jpg')

# Завантаження VGG-19
vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
vgg.trainable = False

# Вибір шарів для змісту та стилю
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Модель для витягнення ознак
outputs = [vgg.get_layer(layer).output for layer in content_layers + style_layers]
feature_extractor = tf.keras.Model(inputs=vgg.input, outputs=outputs)

# Обчислення матриці Грама
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Функція втрат
def compute_loss(generated_img, content_img, style_img):
    content_features = feature_extractor(content_img)
    style_features = feature_extractor(style_img)
    gen_features = feature_extractor(generated_img)
    
    # Втрата змісту
    content_loss = tf.reduce_mean((gen_features[0] - content_features[0]) ** 2)
    
    # Втрата стилю
    style_loss = 0
    for i in range(len(style_layers)):
        gram_style = gram_matrix(style_features[i + len(content_layers)])
        gram_gen = gram_matrix(gen_features[i + len(content_layers)])
        style_loss += tf.reduce_mean((gram_style - gram_gen) ** 2)
    style_loss /= len(style_layers)
    
    # Загальна втрата
    alpha, beta = 1e3, 1e6
    return alpha * content_loss + beta * style_loss

# Оптимізація
generated_img = tf.Variable(content_img, dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=0.01)

images = []
for step in range(200):
    with tf.GradientTape() as tape:
        loss = compute_loss(generated_img, content_img, style_img)
    grads = tape.gradient(loss, generated_img)
    optimizer.apply_gradients([(grads, generated_img)])
    
    if step % 10 == 0:
        print(f"Step {step}, Loss: {loss.numpy()}")
        img = deprocess_img(generated_img.numpy()[0])
        images.append(img)

# Збереження результатів
imageio.mimsave('stylized_animation.gif', images, duration=0.1)
plt.imsave('stylized_image.jpg', images[-1])
