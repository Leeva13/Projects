import { BaseProduct } from '../types/BaseProduct';

export const findProduct = <T extends BaseProduct>(products: T[], id: number): T | undefined => {
    return products.find(product => product.id === id);
};