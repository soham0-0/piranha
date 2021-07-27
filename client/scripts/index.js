$(() => {
  $("#check").click(() => {
    fetch(`https://piranha-api.herokuapp.com/${$("#content").val()}`)
      .then((res) => res.json())
      .then((result) => {
        if (result === "ham") {
          $(".result").text("You're Email is Safe.");
        } else {
          $(".result").text("You're Email is NOT Safe.");
        }
      });
  });
});
