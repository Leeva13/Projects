import { Electronics } from '../types/Electronics';
import { Clothing } from '../types/Clothing';

export const electronics: Electronics[] = [
    {
        id: 1,
        name: "Телефон",
        price: 10000,
        category: 'electronics',
        warranty: "1 рік"
    }
];

export const clothing: Clothing[] = [
    {
        id: 2,
        name: "Футболка",
        price: 500,
        category: 'clothing',
        size: "M",
        material: "Бавовна"
    }
];