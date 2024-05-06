//  Mobile Header Appearance
const hamburger = document.querySelector(".hamburger");
const hamburger_menu = document.querySelector(".hamburger-menu");

hamburger.addEventListener("click", () => {
  hamburger_menu.classList.toggle("hamburger-menu-slide");
  console.log("osama");
});

var swiper = new Swiper('.swiper-container', {
  effect: 'coverflow',
  grabCursor: true,
  centeredSlides: true,
  slidesPerView: 'auto',
  coverflowEffect: {
    rotate: 50,
    stretch: 0,
    depth: 100,
    modifier: 1,
    slideShadows: true,
  },
  loop: true,
  autoplay: {
    delay: 5000,
    disableOnInteraction: false,
  },
  pagination: {
    el: '.swiper-pagination',
    clickable: true,
  },
  navigation: {
    nextEl: '.swiper-button-next',
    prevEl: '.swiper-button-prev',
  },
});