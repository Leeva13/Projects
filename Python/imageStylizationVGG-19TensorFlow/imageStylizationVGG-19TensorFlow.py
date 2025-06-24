import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Функція для завантаження та підготовки зображення
def load_and_process_img(path_to_img):
    img = Image.open(path_to_img).resize((224, 224))
    img = np.array(img)
    img = tf.keras.applications.vgg19.preprocess_input(img)
    return img

# Функція для денормалізації зображення (для відображення)
def deprocess_img(processed_img):
    x = processed_img.copy()
    x += [103.939, 116.779, 123.68]  # Відновлення значень BGR
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Завантаження зображень
content_path = 'content.jpg'  # Шлях до твого зображення змісту
style_path = 'style.jpg'      # Шлях до твого зображення стилю
content_image = load_and_process_img(content_path)
style_image = load_and_process_img(style_path)

# Ініціалізація зображення, яке буде стилізуватися (беремо копію змісту)
generated_image = tf.Variable(content_image, dtype=tf.float32)

# Завантаження моделі VGG-19
model = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
model.trainable = False

# Вибір шарів для змісту та стилю
content_layer = 'block5_conv2'
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

# Функція для отримання активацій
def get_features(image, model):
    layers = [content_layer] + style_layers
    outputs = [model.get_layer(layer).output for layer in layers]
    feature_model = tf.keras.Model(model.input, outputs)
    return feature_model(tf.expand_dims(image, axis=0))

# Функція для обчислення матриці Грама
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

# Функція втрат
def compute_loss(content_features, style_features, generated_features):
    content_weight = 1e3  # Вага змісту
    style_weight = 1e-2   # Вага стилю
    
    # Втрата змісту
    content_loss = tf.reduce_mean((generated_features[0] - content_features[0]) ** 2)
    
    # Втрата стилю
    style_loss = 0
    for gen_feat, style_feat in zip(generated_features[1:], style_features[1:]):
        gen_gram = gram_matrix(gen_feat)
        style_gram = gram_matrix(style_feat)
        style_loss += tf.reduce_mean((gen_gram - style_gram) ** 2)
    style_loss /= len(style_layers)
    
    total_loss = content_weight * content_loss + style_weight * style_loss
    return total_loss

# Отримання ознак для змісту та стилю
content_features = get_features(content_image, model)
style_features = get_features(style_image, model)

# Оптимізація
optimizer = tf.optimizers.Adam(learning_rate=5.0)
iterations = 1000

for i in range(iterations):
    with tf.GradientTape() as tape:
        generated_features = get_features(generated_image, model)
        loss = compute_loss(content_features, style_features, generated_features)
    gradients = tape.gradient(loss, generated_image)
    optimizer.apply_gradients([(gradients, generated_image)])
    if i % 100 == 0:
        print(f'Iteration {i}, Loss: {loss.numpy()}')

# Візуалізація результату
result = deprocess_img(generated_image.numpy())
plt.imshow(result)
plt.axis('off')
plt.title('Стилізоване зображення')
plt.show()

# Збереження результату
Image.fromarray(result).save('stylized_image.jpg')