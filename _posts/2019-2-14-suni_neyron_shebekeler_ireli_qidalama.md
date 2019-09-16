---
layout: post
title: Guess who's back'propagate. - Süni neyron şəbəkə və irəliyə ötürmə alqoritmi.
author: Mammad Hajili
---

Salamlar, aylar oldu bizdən yazı görmədiyiniz, bilirik. Deməli məsələ belə idi ki, bizim saytımız çökdü, problemlər oldu və biz daha sadə bir struktura keçməyə qərar verdik. Artıq yazılara Wordpress'dən yox, Jekyll-Now dizaynı olan sadə bir saytdan davam edəcəyik (bəlkə, ə'lər də xoşunuza gələr bu dəfə). Bu arada, neyronlar və siqmoid funksiya haqda olan iki yazı silinib, onları yenidən bərpa etməyə çalışacam.

<img src="https://raw.githubusercontent.com/DERINtelligence/web/master/images/london_blog.jpg">

Deməli, keçək mətləbə. Bugün süni neyron şəbəkənin strukturuna giriş edəcəyik. Əgər ümumi neyron anlayışına və aktivasiya funksiyaları haqda ümumi anlayışınız varsa, SNŞ-lərin sadə quruluşu sizin üçün çətin olmayacaq. (bundan sonra süni neyron şəbəkələri SNŞ'lə ifadə edəcəyik). Belə ki, mən bu yazıda verəcəyim arxitekturada aktivasiya funksiyasını siqmoid funksiya olaraq qəbul edirəm. Gəlin sadə neyronun siqmoidlə aktivasiyanı xatırlayaq.

<div class="center">
    <img src="https://raw.githubusercontent.com/DERINtelligence/web/master/images/sigma.png"> 
</div>

\begin{eqnarray} 
		z = w \cdot x + b
\tag{1}\end{eqnarray} 

\begin{eqnarray}
        \sigma(z) = \frac{1}{1+e^{-z}}
\tag{2}\end{eqnarray}

Bər. (1)'də $$x$$ siqmoid neyronun giriş dəyərlərini, $$w$$ girişə uyğun ağırlıqları, $$b$$ isə sürüşmə dəyərini ifadə edir. Bər. (2) isə siqmoid aktivasiya funksiyasıdır və onunla neyronun çıxış dəyərini hesablayırıq.

Bizim əsas məqsədimiz bu və ya oxşar(digər aktivasiya funksiyaları istifadə edilən) dizaynda olan neyronlardan birlikdə bir şəbəkə qurmaqdır və məsələ şəbəkədirsə, hətta bəzi hallarda bir neçə qat neyronlar toplusu istifadə etmək lazım ola bilər. Belə ki, bioloji neyron şəbəkənin strukturu əsasında süni neyronların bir-birilərinə riyazi və məntiqi qanunauyğunluqla əlaqələnərək əmələ gətirdiyi riyazi model **süni neyron şəbəkə** adlanır.

Ən sadə SNŞ-ə eyni giriş dəyərləri olan bir neçə perseptronu misal gətirə bilərik. Bu şəbəkənin girişi eynən perseptronda olduğu kimi $$x_1, x_2,\dots, x_n$$, çıxışı isə yalnız bir dəyər yox, bir neçə perseptronun yekun dəyərləridir - $$y_1, y_2, \dots, y_m$$. Belə ki, daha mürəkkəb modellər elə bu sadə modelin üzərində qurulur və əksər hallarda elə bu mürəkkəb modellərdən istifadə olunur. 

Şəbəkədə hər bir hesablama mərhələsi *qat*(ingilis dilli ədəbiyyatlarda *layer*, türk dilli ədəbiyyatlarda *katman*) adlanır. Şəbəkənin giriş dəyərləri birlikdə giriş qatı, çıxışı isə çıxış qatı adlanır. Bura qədər hələ ki, bildiyimiz terminlər və dəyərlərlə qarşılaşdıq. Ancaq bizə *gizli* qalan bir xarakteristikanı hələ vurğulamadım; şəbəkələrdə giriş qatı və çıxış arasında olan hesablama mərhələləri *gizli qatlar* (ing. hidden layers) adlanır. Bir neçə gizli qatı olan neyron şəbəkə isə **çoxqatlı SNŞ** adlanır.

**Əlavə məlumat:** SNŞ-də gizli qatın istifadə edilməsinin riyazi səbəbinə daha ətraflı girmək istəmirəm. Səbəbi bu mövzunun bizim danışdığımız konteksdən bir qədər uzaq olmasıdır. Ancaq bunu qeyd etmək istəyirəm ki, nisbətən sadə (1 və ya 2 gizli qatlı) şəbəkə belə istənilən sərhədlənmiş təyin oblastı olan silsiləvi funksiyanı təxmin edə bilər. Əgər riyazi analiz hissəsi sizə çox maraqlıdırsa, bu [linkdəki](https://towardsdatascience.com/representation-power-of-neural-networks-8e99a383586) yazı yaxşı giriş ola bilər.

Aşağıda gördüyünüz neyron şəbəkə(şəkil EPFL universitetin ["Machine Learning"](https://mlo.epfl.ch/page-157255-en-html/) kursunun materiallarından götürülmüşdür.) hər biri *K* ölçüdə olan *L* ədəd gizli qat, *D* ölçülü giriş qatdan və çıxış qatdan ibarətdir. Şəkildə də gördüyünüz kimi hər hesablama mərhələsinin girişi bir əvvəlki qatın çıxışıdır(və ya şəbəkənin girişidir). Geriyə doğru döngü olmadığından və hər addım irəliyə olduğundan belə modellər ingilis dilli ədəbiyyatlarda *feedforward*, türk dilli ədəbiyyatlarda *ileri beslemeli* şəbəkə olaraq qeyd edilir. <s>Azərbaycan dilində bununla bağlı xüsusi bir mənbə tapa bilmədiyimdən və türk dilində də tərcümə zamanı ingilis dilli terminin hərfi mənası işlədiyindən mən də onu *irəli qidalı* şəbəkə adlandırmağa qərar verdim.(qəribə səslənir, bilirəm, əgər sizin də bir təklifiniz varsa, bizə yazsanız sevinərik.)</s>. Bu yazını paylaşdıqdan sonra bu mövzu ilə bağlı bir neçə rəy aldım, "irəli qıdalı" biraz qəribə səsləndiyindən və hər bir qat öz dəyərini növbəti qata giriş olaraq ötürdüyündən tərcüməni "irəliyə ötürmə"-yə dəyişməyə qərara aldım.

<div class="center">
    <img src="https://raw.githubusercontent.com/DERINtelligence/web/master/images/neuralnetwork.png">
</div>

Deməli, bu irəliyə ötürməli şəbəkədə gizli qat $$l$$-dəki (hansı ki, $$l = 1, \dots, L$$) hər bir neyron özündən bir əvvəlki qatdakı bütün neyronlarla əlaqəlidir. Gəlin, $$l-1$$ qatındakı $$i$$ neyronundan $$l$$ qatındakı $$j$$ neyronuna olan əlaqənin ağırlığını $$w_{i, j}^{(l)}$$ ilə işarə edək. Bu halda $$l$$ qatındakı $$j$$ neyronunun dəyəri olan $$x_j^{(l)}$$-ni aşağıdakı bərabərliklə ifadə edə bilərik.

\begin{eqnarray}
        x_j^{(l)} = \phi(\sum_i w_{i, j}^{(l)} x_i^{(l-1)} + b_j^{(l)})
\tag{3}\end{eqnarray}

Yuxarıda da qeyd etdiyim kimi mən bu yazıda aktivasiya funksiyası($$\phi$$) üçün siqmoid funksiya istifadə edəcəm. Ona görə Bər. 3-də funksiyanı siqmoidlə əvəz edək, $$l$$ qatındakı bütün neyronların dəyərini isə, vektor formasında bir daha yazaq.

\begin{eqnarray}
        z^{(l)} = w^{(l)} x^{(l-1)} + b^{(l)}
\tag{4}\end{eqnarray}

\begin{eqnarray}
        x^{(l)} = \sigma(z^{(l)})
\tag{5}\end{eqnarray}

Neyron şəbəkənin yekun çıxış dəyərini almaq üçün isə şəkildən də gördüyümüz kimi son gizli qatın dəyərindən istifadə edirik. 

\begin{eqnarray}
        y = \sigma(z^{(L)})
\tag{6}\end{eqnarray}

Beləliklə biz *irəliyə ötürmə* alqoritmini bitirdik. Sadə dillə bir daha üzərindən keçsək, deyə bilərik ki, bu alqoritmlə giriş qatından başlayaraq hər dəfə bir sonrakı gizli qatı aktivləşdirib, sonda da çıxış dəyərini hesablıyırıq. İndi burada sual yaranır ki, bütün bu şəbəkənin əsas məqsədi olan təxmin etmə nə qədər dəqiqdir, yaratdığımız model təxmin etmək istədiyimiz dəyərə hansı dərəcədə yaxındır. Biz nəticədə, əlbəttə, təxminimizi minimum zərərlə etmək istəyirik. Bu nöqtədə bizə **maşın öyrənmə** alqoritmləri lazım olacaq. Ancaq bu mövzuya bu yazıda girmək istəmirəm, ona görə də, gəlin burda yavaş-yavaş yekunlaşdıraq. Gələn yazıda artıq maşın öyrənmə alqoritmlərinə, itirmə funksiyalarına giriş edəcəyik. Növbəti yazılarda isə, onların irəliyə ötürməli şəbəkələrə necə uyğunlaşdıra biləcəyimizdən danışacağıq. Mən gələn yazıları yazdığım müddətdə siz xətti cəbr və riyazi analiz kimi mövzulara baxsanız əla olar, çünki, maşın öyrənmə alqoritmlərində riyazi düşüncə önəmlidir. Buraya qədər səbr edib oxuduğunuz üçün təşəkkürlər!
