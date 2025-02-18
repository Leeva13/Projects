interface JQuery {
    counterUp(options?: any): JQuery;
    owlCarousel(options?: any): JQuery;
}

declare namespace bootstrap {
    interface ModalOptions {
        backdrop?: boolean | 'static';
        keyboard?: boolean;
        focus?: boolean;
    }

    class Modal {
        constructor(element: Element, options?: ModalOptions);
        show(): void;
        hide(): void;
    }
}

interface JQuery {
    modal(options?: bootstrap.ModalOptions): JQuery;
}
