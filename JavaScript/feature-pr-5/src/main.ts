import { findProduct } from './functions/findProduct';
import { filterByPrice } from './functions/filterByPrice';
import { addToCart } from './functions/addToCart';
import { calculateTotal } from './functions/calculateTotal';
import { electronics, clothing } from './data/testData';
import { CartItem } from './functions/addToCart';

const phone = findProduct(electronics, 1);
console.log("Знайдений товар:", phone);

const affordableClothing = filterByPrice(clothing, 600);
console.log("Одяг за доступною ціною:", affordableClothing);

let cart: CartItem<any>[] = [];
if (phone) {
    cart = addToCart(cart, phone, 1);
}
const total = calculateTotal(cart);
console.log("Загальна вартість:", total);