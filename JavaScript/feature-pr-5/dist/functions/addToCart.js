export const addToCart = (cart, product, quantity) => {
    const updatedCart = [...cart];
    const existingItem = updatedCart.find(item => item.product.id === product.id);
    if (existingItem) {
        existingItem.quantity += quantity;
    }
    else {
        updatedCart.push({ product, quantity });
    }
    return updatedCart;
};
