export function initDropdownHover(): void {
    function toggleNavbarMethod(): void {
        if ($(window).width()! > 992) {
            $('.navbar .dropdown').on('mouseover', function () {
                $('.dropdown-toggle', this).trigger('click');
            }).on('mouseout', function () {
                $('.dropdown-toggle', this).trigger('click').blur();
            });
        } else {
            $('.navbar .dropdown').off('mouseover').off('mouseout');
        }
    }

    $(document).ready(function () {
        toggleNavbarMethod();
        $(window).resize(toggleNavbarMethod);
    });
}
export {};