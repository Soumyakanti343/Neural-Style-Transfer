/* =========================================================
   IMAGE PREVIEW
========================================================= */

function previewFile(input, previewId) {
    const preview = document.getElementById(previewId);
    const file = input.files[0];

    if (!file) {
        return;
    }

    const reader = new FileReader();
    reader.addEventListener("load", function () {
        preview.innerHTML = `
            <img
                src="${reader.result}"
                alt="Image Preview"
            >
        `;

    }, false);
    reader.readAsDataURL(file);
}


/* =========================================================
   PAGE LOAD
========================================================= */

window.addEventListener("load", function () {

    /* -----------------------------------------------------
       Scroll to result after processing
    ----------------------------------------------------- */

    const resultSection =
        document.getElementById("resultSection");

    if (resultSection) {
        resultSection.scrollIntoView({
            behavior: "smooth"
        });

    }


    /* =====================================================
       NEURAL NETWORK BACKGROUND
    ===================================================== */

    const canvas =
        document.getElementById("neuralCanvas");

    if (!canvas) {
        return;
    }

    const ctx =
        canvas.getContext("2d");

    let width;
    let height;

    let particles = [];

    let mouse = {
        x: null,
        y: null,
        radius: 150
    };


    /* -----------------------------------------------------
       Canvas Resize
    ----------------------------------------------------- */

    function resize() {
        width =
            canvas.width =
            window.innerWidth;

        height =
            canvas.height =
            window.innerHeight;
    }


    /* =====================================================
       PARTICLE CLASS
    ===================================================== */

    class Particle {
        constructor() {

            this.x =
                Math.random() * width;

            this.y =
                Math.random() * height;

            this.vx =
                (Math.random() - 0.5) * 0.5;

            this.vy =
                (Math.random() - 0.5) * 0.5;

            this.size =
                Math.random() * 3 + 1.5;

            this.hue =
                Math.floor(Math.random() * 10) * 36;

            this.density =
                Math.random() * 30 + 1;
        }


        /* -------------------------------------------------
           Update Particle
        ------------------------------------------------- */

        update() {
            this.x += this.vx;
            this.y += this.vy;

            /* Mouse Interaction */
            if (mouse.x !== null) {
                const dx =
                    mouse.x - this.x;

                const dy =
                    mouse.y - this.y;

                const distance =
                    Math.sqrt(
                        dx * dx +
                        dy * dy
                    );


                if (
                    distance < mouse.radius &&
                    distance !== 0
                ) {

                    const forceDirectionX =
                        dx / distance;

                    const forceDirectionY =
                        dy / distance;

                    const force =
                        (mouse.radius - distance) /
                        mouse.radius;

                    const directionX =
                        forceDirectionX *
                        force *
                        this.density;

                    const directionY =
                        forceDirectionY *
                        force *
                        this.density;

                    this.x -= directionX;
                    this.y -= directionY;
                }
            }


            /* Boundary Collision */

            if (
                this.x < 0 ||
                this.x > width
            ) {
                this.vx *= -1;
            }

            if (
                this.y < 0 ||
                this.y > height
            ) {
                this.vy *= -1;
            }


            /* Color Animation */
            this.hue =
                (this.hue + 0.2) % 360;
        }


        /* -------------------------------------------------
           Draw Particle
        ------------------------------------------------- */

        draw() {

            ctx.fillStyle =
                `hsla(
                    ${this.hue},
                    70%,
                    60%,
                    0.3
                )`;

            ctx.beginPath();

            ctx.arc(
                this.x,
                this.y,
                this.size,
                0,
                Math.PI * 2
            );

            ctx.fill();
        }
    }


    /* =====================================================
       INITIALIZE PARTICLES
    ===================================================== */

    function initParticles() {
        particles = [];

        const numParticles =
            Math.floor(
                window.innerWidth / 15
            );

        for (
            let i = 0;
            i < numParticles;
            i++
        ) {

            particles.push(
                new Particle()
            );
        }
    }


    /* =====================================================
       ANIMATION LOOP
    ===================================================== */

    function animate() {
        ctx.clearRect(
            0,
            0,
            width,
            height
        );


        particles.forEach(
            (particle, index) => {
                particle.update();
                particle.draw();


                /* Connect nearby particles */

                for (
                    let j = index + 1;
                    j < particles.length;
                    j++
                ) {

                    const particle2 =
                        particles[j];

                    const distance =
                        Math.hypot(
                            particle.x -
                            particle2.x,

                            particle.y -
                            particle2.y
                        );


                    if (distance < 100) {
                        ctx.strokeStyle =
                            `hsla(
                                ${particle.hue},
                                70%,
                                60%,
                                ${
                                    0.15 *
                                    (1 - distance / 100)
                                }
                            )`;

                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(
                            particle.x,
                            particle.y
                        );
                        ctx.lineTo(
                            particle2.x,
                            particle2.y
                        );
                        ctx.stroke();
                    }
                }
            }
        );


        requestAnimationFrame(
            animate
        );
    }


    /* =====================================================
       WINDOW EVENTS
    ===================================================== */

    window.addEventListener(
        "resize",
        function () {

            resize();

            initParticles();

        }
    );


    window.addEventListener(
        "mousemove",
        function (event) {
            mouse.x =
                event.clientX;

            mouse.y =
                event.clientY;
        }
    );


    /* =====================================================
       START NEURAL BACKGROUND
    ===================================================== */

    resize();
    initParticles();
    animate();


    /* =====================================================
       RANGE SLIDER
    ===================================================== */

    const range =
        document.getElementById(
            "alphaRange"
        );

    const rangeValue =
        document.getElementById(
            "rangeValue"
        );


    function updateRange() {

        if (!range || !rangeValue) {
            return;
        }

        const val =
            range.value;

        const min =
            range.min || 0;

        const max =
            range.max || 1;


        const percentage =
            ((val - min) * 100) /
            (max - min);


        /* Update displayed value */

        rangeValue.textContent =
            val;


        /* Position value bubble */

        const thumbWidth = 24;
        const offset =
            thumbWidth / 2 -
            (percentage * thumbWidth / 100);

        rangeValue.style.left =
            `calc(
                ${percentage}% +
                ${offset}px
            )`;


        /* Update slider gradient */

        range.style.background =
            `linear-gradient(
                to right,
                #818cf8 0%,
                #c084fc ${percentage}%,
                rgba(30, 41, 59, 0.8)
                ${percentage}%,
                rgba(30, 41, 59, 0.8)
                100%
            )`;
    }


    if (range) {
        range.addEventListener(
            "input",
            updateRange
        );

        updateRange();
    }

});




