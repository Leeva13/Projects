import { BaseProduct } from '../types/BaseProduct';
import { CartItem } from './addToCart';

export const calculateTotal = <T extends BaseProduct>(cart: CartItem<T>[]): number => {
    return cart.reduce((total, item) => total + item.product.price * item.quantity, 0);
};