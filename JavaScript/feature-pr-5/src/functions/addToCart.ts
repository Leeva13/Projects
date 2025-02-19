import { BaseProduct } from '../types/BaseProduct';

export type CartItem<T> = {
    product: T;
    quantity: number;
};

export const addToCart = <T extends BaseProduct>(cart: CartItem<T>[], product: T, quantity: number): CartItem<T>[] => {
    const updatedCart = [...cart];
    const existingItem = updatedCart.find(item => item.product.id === product.id);

    if (existingItem) {
        existingItem.quantity += quantity;
    } else {
        updatedCart.push({ product, quantity });
    }

    return updatedCart;
};