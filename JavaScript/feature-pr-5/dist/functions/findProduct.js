export const findProduct = (products, id) => {
    return products.find(product => product.id === id);
};
