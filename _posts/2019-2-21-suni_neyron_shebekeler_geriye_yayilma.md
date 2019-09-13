---
layout: post
title: Guess who's back'propagate. - Zəncir qaydası və geriyə yayılma.
---
<style type="text/css">
.center {
  display: block;
  margin-left: auto;
  margin-right: auto;
  width: 50%
}
</style>

Salamlar, keçən [yazıda](http://derintelligence.az/suni_neyron_shebekeler_mashin_oyrenme/) maşın öyrənmənin təməlini təşkil edən xəta funksiyalarından və ən önəmli alqoritmlərdən olan nöqtəvi meyilli azalma haqqında danışmışdıq. Bu dəfə istəyirəm, keçən dəfə qeyd etdiklərimin süni neyron şəbəkələrdə necə istifadə edildiyi haqqında danışım. Belə ki, bu yazı nöqtəvi meyilli azalmanın neyron şəbəkədə tətbiqi olan geriyə yayılma və onun həlli üçün istifadə edəcəyimiz riyazi metod olan zəncir qaydası ilə bağlıdır.

<img src="https://raw.githubusercontent.com/DERINtelligence/web/master/images/lausanne.jpg" style="width:700px;height:400px">

Gəlin ilk olaraq neyron şəbəkə arxitekturasını və irəliyə ötürmə qaydasını yada salaq.

<img src="https://raw.githubusercontent.com/DERINtelligence/web/master/images/neuralnetwork.png" style="width:700px;height:400px">

Qeyd: Bu yazıda xəta funksiyası ($$L$$), və son qat ($$L$$) qarışmasın deyə, xəta funksiyasını $$C$$ ilə əvəz edəcəm.

\begin{eqnarray}
        z^{(l)} = w^{(l)} x^{(l-1)} + b^{(l)}
\tag{1}\end{eqnarray}

\begin{eqnarray}
        x^{(l)} = \sigma(z^{(l)})
\tag{2}\end{eqnarray}

İndi isə xəta funksiyası və stoxastik nöqtəvi meyilli azalmanı xatırlayaq.

\begin{eqnarray}
		C = \frac{1}{2}{(Y - f)} ^ 2
\tag{3}\end{eqnarray}

Süni neyron şəbəkədə riyazi modelin çıxış dəyəri şəbəkənin çıxış dəyərinə bərabərdir. Ona görə də Bər. 3-dəki $$f$$ funksiyası $$x_{(L)}$$ - ə bərabər olacaqdır. Bu vəziyyətdə Bər. 3-ü belə yaza bilərik:

\begin{eqnarray}
		C = \frac{1}{2}{(Y - x^{(L)})} ^ 2
\tag{4}\end{eqnarray}

Bu xəta funksiyasına uyğun nöqtəvi meyilli azalma isə ağırlıq və sürüşmə əmsalının hər bir iterasiyada dəyişməsindən ibarətdir.

\begin{eqnarray}
		w^{(t+1)} = w^{(t)} - \beta_1\frac{\partial C}{\partial w^{(t)}}
\tag{5}\end{eqnarray}

\begin{eqnarray}
		b^{(t+1)} = b^{(t)} - \beta_2\frac{\partial }{\partial b^{(t)}}
\tag{6}\end{eqnarray}

Bər. 4-də gördüyümüz kimi biz xəta dəyərini hesablayarkən şəbəkənin çıxış dəyəri olan $$x^{(L)}$$ - in verilən toplusundakı cavab ilə kvadratik fərqini hesablayırıq. $$x^{(L)}$$ $$z^{(L)}$$-dən(Bər 2.), $$z_{(L)}$$ isə öz növbəsində ağırlıq və sürüşmə dəyərlərindən(Bər 1.) asılıdır. Yəni $$x^{(L)}$$ $$w^{(L)}$$ və $$b^{(L)}$$-dən asılıdır. Ancaq burda önəmli bir sual yaranır, xətanı azaltmaq üçün Bər 5. və Bər 6. yalnızca $$w^{(L)}$$ və $$b^{(L)}$$ - ə tətbiq etməliyik?

Belə olduğu halda şəbəkənin gizli qatlarına uyğun ağırlıq və sürüşmə əmsallarını yox saymış olacağıq. Ancaq Bər 1.-də də gördüyümüz kimi hər qatın dəyəri özündən əvvəlki qatın dəyərindən də asılıdır, yəni ağırlıqlarda və sürüşmədə uyğun dəyişiklik edilməsə, çıxış qatının dəyəri tam optimal olmayacaq. Problem ondadır ki, biz xəta dəyərinin yalnızca çıxış qatında hesablaya bilirik. Bəs onda biz gizli qatlardakı parametrlərində nəyə uyğun dəyişliklər edəcəyik? Başqa bir sözlə desək, nöqtəvi meyilli azalmanı gizli qatları necə tətbiq etmək olar?

Şəbəkədə hər qat özündən bir əvvəlki qatdan asılı olduğundan, çıxış qatınının dəyərini təyin oblastı giriş qatı olan mürəkkəb qeyri-xətti funksiya hesab edə bilərik. Yəni şəbəkənin əsas irəliyə ötürmə prinsipini açılmış şəkildə belə yaza bilərik.

$$\begin{equation}
\begin{split}
   z^{(1)} &= w^{(1)} x^{(0)} + b^{(1)} \\
   x^{(1)} &= \sigma(z^{(1)}) \\
   z^{(2)} &= w^{(2)} x^{(1)} + b^{(2)} \\
   x^{(2)} &= \sigma(z^{(2)}) \\
   & \dots \\
   z^{(l)} &= w^{(l)} x^{(l-1)} + b^{(l)} \\
   x^{(l)} &= \sigma(z^{(l)}) \\
   & \dots \\
   z^{(L)} &= w^{(L)} x^{(L-1)} + b^{(L)} \\
   x^{(L)} &= \sigma(z^{(L)}) \\
\end{split}
\tag{7}\end{equation}$$

Çıxış dəyərinin mürəkkəb funksiya olması bizə hər bir qat üçün nöqtəvi meyili hesablayarkən çox kömək olacaq. Belə ki, biz differensialları hesablayarkən mürəkkəb funksiyaya tətbiq olunan ən önəmli qaydalardan olan **zəncir qaydasından** istifadə edəcəyik. Bu qayda ilə, yəqin ki, oxuyucularımızın əksəriyyəti tanışdır, ancaq, gəlin, qısaca xatırlayaq.

# Zəncir Qaydası

İxtiyari $$f$$, $$g$$ və $$F$$ funksiyasıları üçün, $$F=f(g(x))$$ qaydasını qəbul edək. Məqsədimiz $$F$$-in $$x$$ parametrinə uyğun törəməsini hesablamaqdır. Gəlin,  
$$g(x)$$ - i $$y$$, $$f(y)$$ - i isə $$z$$ ilə işarə edək. Bu halda zəncir qaydası aşağıdakı kimidir:

$$\begin{equation}
\begin{split}
   F^\prime &= \frac{dz}{dx} \\
   \frac{dz}{dx} &= \frac{dz}{dy} \frac{dy}{dx} \\
   \frac{dz}{dx} &= f^\prime(g(x)) g^\prime(x) \\
   F^\prime &= \frac{dz}{dx} =  f^\prime(g(x)) g^\prime(x)\\
\end{split}
\tag{8}\end{equation}$$


İndi isə qayıdaq əsas problemə - nöqtəvi meyillərin hesablanmasına. Bizim məqsədimiz hər bir qat üçün nöqtəvi meyillər - $$\frac{\partial C}{\partial w^{(t)}}$$ və $$\frac{\partial C}{\partial b^{(t)}}$$ - i hesablamaqdır. İlk addım olaraq bu hesablamaları asanlaşdırmaq üçün əlavə bir parametrdən - $$\delta$$ -dan istifadə edəcəyik.

\begin{eqnarray}
  \delta^l = \frac{\partial C}{\partial z^l}, \forall l \in [1, L]
\tag{9}\end{eqnarray}


Yuxarıda vurğuladığım kimi nöqtəvi meyil üçün gərəkli olan xəta funskiyasının dəyəri çıxış qatının dəyərindən asılıdır. Buna görə də, ilk olaraq bu qata uyğun meylin hesablamasından başlayırıq. Mən buradakı hesablamalar zamanı nəticələrin həll yolunu da izah edəcəyəm, əgər "riyaziyyatı boş ver, mənə cavabı ver" deyirsinizsə, hər bərabərliyin son nəticəsinə baxmağınız kifayətdir.

İlk olaraq çıxış qatında başlayırıq.

\begin{eqnarray}
  \delta^L &= \frac{\partial C}{\partial z^L_j} \\
\tag{9}\end{eqnarray}

Növbəti addım kimi bu bərabərliyə zəncir qaydasını tətbiq edirik.

\begin{eqnarray}
  \delta^L &= \frac{\partial C}{\partial x^L} \frac{\partial x^L}{\partial z^L}
\tag{11}\end{eqnarray}

$$x^L = \sigma(z^L)$$ bərabərliyini xatırlasaq, yuxarıdakı bərabərliyin ikinci hissəsini $$\sigma'(z^L_j)$$ kimi yaza bilərik.

\begin{eqnarray}
  \delta^L = \frac{\partial C}{\partial x^L} \sigma'(z^L_j)
\tag{12}\end{eqnarray}

Diqqət etsək, görə bilərik ki, bərabərliyin birinci hissəsi isə Bər 4. - ün differensialıdır. Buna görə ifadəni belə yaza bilərik.

\begin{eqnarray} 
  \delta^L = (x^L-y) \sigma'(z^L).
\tag{13}\end{eqnarray}


$$\delta^L$$ şəbəkənin **çıxış itkisi**(ing. output error) adlanır. Növbəti addımlarda məqsədimiz bu itkinin gizli qatlara yayılmasını təmin etməkdir. Ümumi ideya bundan ibarətdir ki, hər bir qatdakı itkini ondan əvvəlki qatdakı itkini hesablamaq üçün istifadə edəcəyik. Bu rekursiv məntiq səbəbi ilə istifadə etdiyimiz bu alqoritm itkinin geriyə yayılması adlanır. Hər qatda hesablayacağımız bu itkinin köməkliyi ilə ümumi xətanın hər qatdakı ağırlıq və sürüşməyə nəzərən olan meyilli azalmasını tapa biləcəyik. Beləliklə, bu bizə hər qatdakı ağırlıq və sürüşməni yeniləməyə və optimal həlli tapmağa imkan verəcək.

Növbəti addımda isə ixtiyari gizli qat $$l$$ üçün itki - $$\delta^l$$ - in hesablanmasına baxaq.

\begin{eqnarray}
  \delta^l &= \frac{\partial C}{\partial z^l_j} \\
\tag{14}\end{eqnarray}

Çıxış qatından fərqli olaraq gizli qat bir neyron yox, bir neçə neyrondan ibarət ola bilər. $$l$$ qatındakı hər bir $$j$$ neyronu tam əlaqələnmiş(ing. fully connected) neyron şəbəkədə $$l+1$$ qatındakı bütün neyronlarla bağlanmışdır. Buna görə də Bər. 13-ü zəncir qaydasından istifadə edərək, hər bir $$\delta^l_j$$ üçün belə yaza bilərik.

$$\begin{eqnarray}
  \begin{split}
    \delta^l_j &= \frac{\partial C}{\partial z^l_j} \\
    &= \sum_k \frac{\partial C}{\partial z^{l+1}_k} \frac{\partial z^{l+1}_k}{\partial z^l_j} \\
    &= \sum_k \delta^{l+1}_k \frac{\partial z^{l+1}_k}{\partial z^l_j}
  \end{split}
\tag{15}\end{eqnarray}$$

İfadənin ikinci hissəsini hesablamaq üçün $$l+1$$ qatı üçün irəliyə ötürmə qaydasından istifadə edə bilərik.

$$\begin{eqnarray}
  z^{l+1}_k = \sum_j w^{l+1}_{kj} x^l_j +b^{l+1}_k = \sum_j w^{l+1}_{kj} \sigma(z^l_j) +b^{l+1}_k.
\tag{16}\end{eqnarray}$$


Buna uyğun differensialı həll edə bilərik:

$$\begin{eqnarray}
  \frac{\partial z^{l+1}_k}{\partial z^l_j} = w^{l+1}_{kj} \sigma'(z^l_j).
\tag{17}\end{eqnarray}$$

Hər bir şeyi birləşdirdikdə, Bər 13-ü belə yaza bilərik

$$\begin{eqnarray}
  \delta^l_j = \sum_k w^{l+1}_{kj}  \delta^{l+1}_k \sigma'(z^l_j).
\tag{18.1}\end{eqnarray}$$


Bu bərabərliyi vektor formasında belə də yaza bilərik

$$\begin{eqnarray} 
  \delta^l = ((w^{l+1})^T \delta^{l+1}) \odot \sigma'(z^l),
\tag{18.2}\end{eqnarray}$$

Burada $$\odot$$ elementvari və ya [Hadamart](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) hasilini ifadə edir.

Artıq $$\delta^l, l = 1, 2, \dots, L$$ bildiyimiz üçün onlardan istifadə edərək $$\frac{\partial C}{\partial w}$$ və $$\frac{\partial C}{\partial b}$$ həll edə bilərik.


$$\begin{eqnarray}
  \frac{\partial C}{\partial w^l_{jk}} &= \frac{\partial C}{\partial z^l_{j}}\frac{\partial z^l_{j}}{\partial w^l_{jk}} \\
  &= \delta^l_j x^{l-1}_k
\tag{19.1}\end{eqnarray}$$


$$\begin{eqnarray}
  \frac{\partial C}{\partial b^l_{j}} &= \frac{\partial C}{\partial z^l_{j}}\frac{\partial z^l_{j}}{\partial b^l_{j}} \\
  &= \delta^l_j
\tag{19.2}\end{eqnarray}$$

Qeyd: Arada bəzi sadə törəmə əməliyyatlarını sizin incələməyiniz üçün qəsdən buraxdım.

Yekun olaraq bütün geriyə yayılma alqoritminin elementlərini birlikdə yazaq:

1. İrəliyə ötürmə: Hər bir $$l = 2, 3, \ldots, L$$ üçün $$z^l$$ və $$x^l$$ - i hesablamaq.
2. Çıxış itkisi: $$\delta^L$$ - i hesablamaq.
3. İtkini geriyə yayma: Hər bir $$l = L-1, L-2, \ldots, 2$$ üçün $$\delta$$ - i hesablamaq
4. Nöqtəvi meyilləri hesablama: Xəta funksiyasının ağırlıq və sürüşməyə əsasən nöqtəvi meyillərini $$\frac{\partial C}{\partial w}$$ və $$\frac{\partial C}{\partial b}$$  hesablamaq.

