## Subresource Integrity

If you are loading Highlight.js via CDN you may wish to use [Subresource Integrity](https://developer.mozilla.org/en-US/docs/Web/Security/Subresource_Integrity) to guarantee that you are using a legimitate build of the library.

To do this you simply need to add the `integrity` attribute for each JavaScript file you download via CDN. These digests are used by the browser to confirm the files downloaded have not been modified.

```html
<script
  src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.12.0/highlight.min.js"
  integrity="sha384-KnPvYPx1poT554tHDV1nuYV9sOkh4cZPBvLZQlXgJmoRQZPdgQNwL50/xq9kynp9"></script>
<!-- including any other grammars you might need to load -->
<script
  src="//cdnjs.cloudflare.com/ajax/libs/highlight.js/11.12.0/languages/go.min.js"
  integrity="sha384-orYKHAs3chK3oDMQLy5ywrzoY8z9zvzfmNIjmVxKXioAUtwDhP+xf6THWYSI/43Y"></script>
```

The full list of digests for every file can be found below.

### Digests

```
sha384-A3L5Tf7tUTEvvb7CBQ7/Ss1TL493yDjIre9lL9+hewzsmlGvDaF0zr2jtRL/as9+ /es/languages/c.js
sha384-QjLTklOpGRj5BiUGiD7/7RYVgazp8V+oQvLPmM+goByoChSDDxbVf7ztjZ5MtHO2 /es/languages/c.min.js
sha384-jJSKb0kbVcDJLL6D9Zc80g+XBwa84Ogoi6JH0dZMQJlbzn6ZCjYFE2WtU6oTc8Au /es/languages/cpp.js
sha384-p1ClIV57O7/nb8vDAVFIhZ7u0/8oonjMxc4Yd8EarzxmLPMiQAV95I0ogb+ZkEFN /es/languages/cpp.min.js
sha384-lt0gg86v1uEAI7/c2402dN+9T2iQXopS6OJ8P1jHo1uOZu1zIV5sX8vmlOLGozq0 /es/languages/json.js
sha384-UJwfLbfKiYs+crNOV1xJL6wDet7JiH/Kav6qZ9c+cOjar4piP1X0VQqwVE3RJt+z /es/languages/json.min.js
sha384-M29KVshhb1nOr+V9fD7zculEkusgFmZdZpLoQxEiEJ7nTncEiXzJ9/tXuofiOIYK /es/languages/python.js
sha384-1SN2ySEZrgpb2XQB4PRJli/u1Jj5sm1j+Be77jnSNNmKzhUZK8OBkGFFexuOPSHu /es/languages/python.min.js
sha384-++jr2IZH8qZUHclpwLKPdPB67oloAwlp6bVn6IOBm1PRyrATGwIJDemLwIjTXWhI /languages/c.js
sha384-XUlYYDNkxUkC5F86/mGCoT9eEdPUigCR+8ihyrqFat6tMD0/GRHkorEtHKgBSS7D /languages/c.min.js
sha384-Fu6emj6YokTl+GgIHZRr4tqDnAWVGmEP67A0XguFYxkNQLhiKEgZa3J2t21BQ7Fq /languages/cpp.js
sha384-csK05VfEiSG0s/zJz067fB5PZCE5+Uo1yqgoa1zZkSG1cmBaOuXEGSbVr2YqAghu /languages/cpp.min.js
sha384-xRs5pKapNPranWV1tpWwbWD8FN6u6gwlBXWwW+3wcfgCrXtc+VuTjt6Ff2MZTUoZ /languages/json.js
sha384-BuKQB2q4LIWSYzKqO0qkAQOdYLvqUmSmzUtLZDkTHy5po4tY7DSEcu/r5555QON2 /languages/json.min.js
sha384-vHFyigyrxXKrLcfzhSiSgjFvq0h4IaawkRiHBKD7g4xx6ny6hQvUeIe5o0vcr2Qi /languages/python.js
sha384-MjEDnJ45PjESeDQtPkC0gdCZlwA6MWEYLl/Qw95D9e1Ow8yAlz/xkS/EDIxFiig6 /languages/python.min.js
sha384-/HkboZYcEkW74o1st2mASckLcE567d4qCF8xqMUeIApI6CvHUqWkI3J/VtjCVMzt /highlight.js
sha384-++I4jAYk/T+s7T4bhSQgK4BIzvEApAAms25Bi5p1Kl7mlhl4S6UYxIJcAEJMCA1s /highlight.min.js
```

