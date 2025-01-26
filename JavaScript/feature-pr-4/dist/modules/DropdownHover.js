export function initDropdownHover() {
    function toggleNavbarMethod() {
        if ($(window).width() > 992) {
            $('.navbar .dropdown').on('mouseover', function () {
                $('.dropdown-toggle', this).trigger('click');
            }).on('mouseout', function () {
                $('.dropdown-toggle', this).trigger('click').blur();
            });
        }
        else {
            $('.navbar .dropdown').off('mouseover').off('mouseout');
        }
    }
    $(document).ready(function () {
        toggleNavbarMethod();
        $(window).resize(toggleNavbarMethod);
    });
}
//# sourceMappingURL=DropdownHover.js.map